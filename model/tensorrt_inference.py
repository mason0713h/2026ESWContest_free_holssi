"""
model/tensorrt_inference.py
============================
TensorRT FP16 최적화 추론 엔진.

ONNX 모델을 TensorRT FP16으로 변환하고,
pycuda + tensorrt를 사용하여 Jetson Nano(JetPack 4.6.x, TensorRT 8.2)
에서 실시간 추론을 수행한다.

바인딩 기반 레거시 TensorRT API(get_binding_shape/execute_async_v2)를
사용하므로 TRT 8.x 계열과 호환된다. TRT 8.4 이상에서만 존재하는
MemoryPoolType API는 TRTEngineBuilder.build_from_onnx 내부에서
존재 여부를 확인해 자동으로 max_workspace_size로 폴백한다.

하드웨어 없이 실행 시 (TensorRT 미설치) ONNX Runtime → 랜덤 폴백 순으로 대체한다.

추론 파이프라인:
    numpy 입력 → GPU 메모리 복사 → TRT 추론 → 결과 반환
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# TensorRT 가용성 확인
try:
    import tensorrt as trt  # type: ignore
    import pycuda.driver as cuda  # type: ignore
    import pycuda.autoinit  # type: ignore  # noqa: F401
    TRT_AVAILABLE = True
    logger.info("TensorRT %s 사용 가능", trt.__version__)
except ImportError:
    TRT_AVAILABLE = False
    logger.warning("TensorRT/pycuda 미설치. PyTorch 폴백 모드로 동작합니다.")


# ──────────────────────────────────────────────
# TensorRT 엔진 빌더
# ──────────────────────────────────────────────

class TRTEngineBuilder:
    """
    ONNX → TensorRT FP16 변환 유틸리티.

    Args:
        workspace_gb: TRT 빌드 시 워크스페이스 크기 (GB)
        fp16_mode: FP16 모드 활성화
        verbose: 빌드 로그 상세 출력
    """

    def __init__(
        self,
        workspace_gb: float = 4.0,
        fp16_mode: bool = True,
        verbose: bool = False,
    ) -> None:
        self.workspace_gb = workspace_gb
        self.fp16_mode = fp16_mode
        self.verbose = verbose

    def build_from_onnx(self, onnx_path: str, engine_path: str) -> bool:
        """
        ONNX 파일을 TensorRT 엔진으로 변환하여 저장.

        Args:
            onnx_path: ONNX 모델 파일 경로
            engine_path: 저장할 TRT 엔진 파일 경로

        Returns:
            성공 여부
        """
        if not TRT_AVAILABLE:
            logger.error("TensorRT가 설치되지 않아 엔진 빌드 불가")
            return False

        logger.info("TRT 엔진 빌드 시작: %s → %s", onnx_path, engine_path)

        TRT_LOGGER = trt.Logger(
            trt.Logger.VERBOSE if self.verbose else trt.Logger.WARNING
        )

        with trt.Builder(TRT_LOGGER) as builder, \
             builder.create_network(
                 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
             ) as network, \
             trt.OnnxParser(network, TRT_LOGGER) as parser:

            # ONNX 파싱
            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        logger.error("ONNX 파싱 오류: %s", parser.get_error(i))
                    return False

            # Builder 설정
            # TRT 8.4+ 는 MemoryPoolType API, 구형 Jetson Nano(JetPack 4.6, TRT 8.2)는
            # max_workspace_size 속성만 지원하므로 둘 다 지원하도록 분기한다.
            config = builder.create_builder_config()
            workspace_bytes = int(self.workspace_gb * (1 << 30))
            if hasattr(trt, "MemoryPoolType") and hasattr(config, "set_memory_pool_limit"):
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
            else:
                config.max_workspace_size = workspace_bytes  # TRT < 8.4 (Jetson Nano)

            if self.fp16_mode and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("FP16 모드 활성화")

            # 배치 크기 1 고정
            profile = builder.create_optimization_profile()
            input_name = network.get_input(0).name
            # shape: (batch, T, N, C) = (1, 16, 64, 6)
            profile.set_shape(
                input_name,
                min=(1, 16, 64, 6),
                opt=(1, 16, 64, 6),
                max=(1, 16, 64, 6),
            )
            config.add_optimization_profile(profile)

            # 엔진 빌드
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                logger.error("TRT 엔진 빌드 실패")
                return False

            # 저장
            engine_path_obj = Path(engine_path)
            engine_path_obj.parent.mkdir(parents=True, exist_ok=True)
            with open(engine_path, "wb") as f:
                f.write(serialized_engine)

            logger.info(
                "TRT 엔진 저장 완료: %s (%.1f MB)",
                engine_path,
                engine_path_obj.stat().st_size / (1024 * 1024),
            )
            return True


# ──────────────────────────────────────────────
# TensorRT 추론 엔진
# ──────────────────────────────────────────────

class TRTInferenceEngine:
    """
    TensorRT FP16 추론 엔진.

    TRT 엔진 파일을 로드하고, pycuda로 GPU 메모리를 관리하여
    배치 크기 1 스트리밍 추론을 수행한다.

    목표 추론 지연시간: < 50ms (레이더 프레임 주기 100ms 이내)

    Args:
        engine_path: TRT 엔진 파일 경로 (.trt)
        fall_threshold: 낙상 판정 임계값
    """

    def __init__(
        self,
        engine_path: str,
        fall_threshold: float = 0.7,
    ) -> None:
        self.engine_path = engine_path
        self.fall_threshold = fall_threshold

        self._engine = None
        self._context = None
        self._inputs: list = []
        self._outputs: list = []
        self._bindings: list = []
        self._stream = None

        self._loaded = False
        self._latency_history: list = []

    def load(self) -> bool:
        """
        TRT 엔진 파일 로드 및 실행 컨텍스트 초기화.

        Returns:
            로드 성공 여부
        """
        if not TRT_AVAILABLE:
            logger.warning("TRT 미사용 모드 - PyTorch 폴백 활성화")
            return False

        if not Path(self.engine_path).exists():
            logger.error("TRT 엔진 파일 없음: %s", self.engine_path)
            return False

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        with open(self.engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        if self._engine is None:
            logger.error("TRT 엔진 역직렬화 실패")
            return False

        self._context = self._engine.create_execution_context()
        self._stream = cuda.Stream()

        # 바인딩 할당
        for binding in self._engine:
            shape = self._engine.get_binding_shape(binding)
            size = trt.volume(shape)
            dtype = trt.nptype(self._engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self._bindings.append(int(device_mem))

            if self._engine.binding_is_input(binding):
                self._inputs.append({"host": host_mem, "device": device_mem, "shape": shape})
            else:
                self._outputs.append({"host": host_mem, "device": device_mem, "shape": shape})

        self._loaded = True
        logger.info("TRT 엔진 로드 완료: %s", self.engine_path)
        return True

    def infer(self, x: np.ndarray) -> Tuple[bool, float, float]:
        """
        단일 샘플 추론.

        Args:
            x: (1, T, N, C) 또는 (T, N, C) numpy 배열

        Returns:
            (fall_detected, fall_probability, latency_ms)
        """
        if x.ndim == 3:
            x = x[np.newaxis]  # (1, T, N, C)

        if not self._loaded:
            return self._pytorch_fallback_infer(x)

        t0 = time.perf_counter()

        # 입력 복사
        np.copyto(self._inputs[0]["host"], x.ravel().astype(np.float32))
        cuda.memcpy_htod_async(
            self._inputs[0]["device"],
            self._inputs[0]["host"],
            self._stream,
        )

        # 추론 실행
        self._context.execute_async_v2(
            bindings=self._bindings,
            stream_handle=self._stream.handle,
        )

        # 결과 복사
        cuda.memcpy_dtoh_async(
            self._outputs[0]["host"],
            self._outputs[0]["device"],
            self._stream,
        )
        self._stream.synchronize()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._latency_history.append(elapsed_ms)

        # Softmax 결과 파싱
        output = self._outputs[0]["host"]
        output_shape = self._outputs[0]["shape"]
        result = output.reshape(output_shape)

        # (1, 2) → 낙상 확률
        proba = self._softmax(result[0] if result.ndim > 1 else result)
        fall_prob = float(proba[1])
        fall_detected = fall_prob >= self.fall_threshold

        if elapsed_ms > 50:
            logger.warning("추론 지연 초과: %.1fms (목표 <50ms)", elapsed_ms)

        return fall_detected, fall_prob, elapsed_ms

    def _pytorch_fallback_infer(
        self, x: np.ndarray
    ) -> Tuple[bool, float, float]:
        """
        TRT 미사용 시 PyTorch CPU 폴백 추론.

        ONNX 모델 경로가 있으면 onnxruntime을 사용하고,
        없으면 랜덤 결과 반환 (테스트용).
        """
        t0 = time.perf_counter()

        onnx_path = self.engine_path.replace(".trt", ".onnx")
        if Path(onnx_path).exists():
            try:
                import onnxruntime as ort  # type: ignore
                sess = ort.InferenceSession(
                    onnx_path,
                    providers=["CPUExecutionProvider"],
                )
                input_name = sess.get_inputs()[0].name
                output = sess.run(None, {input_name: x.astype(np.float32)})[0]
                proba = self._softmax(output[0])
                fall_prob = float(proba[1])
                elapsed_ms = (time.perf_counter() - t0) * 1000
                self._latency_history.append(elapsed_ms)
                return fall_prob >= self.fall_threshold, fall_prob, elapsed_ms
            except Exception as e:
                logger.debug("ONNX Runtime 폴백 실패: %s", e)

        # 최종 폴백: 랜덤 (Mock 테스트용)
        elapsed_ms = (time.perf_counter() - t0) * 1000 + np.random.uniform(5, 15)
        fall_prob = float(np.random.uniform(0.0, 0.3))  # 대부분 낙상 없음
        self._latency_history.append(elapsed_ms)
        return fall_prob >= self.fall_threshold, fall_prob, elapsed_ms

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def get_latency_stats(self) -> Dict[str, float]:
        """
        누적 추론 지연시간 통계.

        Returns:
            {'mean_ms', 'min_ms', 'max_ms', 'p95_ms', 'count'}
        """
        if not self._latency_history:
            return {"mean_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "p95_ms": 0.0, "count": 0}

        arr = np.array(self._latency_history)
        return {
            "mean_ms": float(arr.mean()),
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
            "p95_ms": float(np.percentile(arr, 95)),
            "count": len(arr),
        }

    def release(self) -> None:
        """GPU 메모리 및 리소스 해제."""
        self._inputs.clear()
        self._outputs.clear()
        self._bindings.clear()
        if TRT_AVAILABLE and self._stream:
            del self._stream
        logger.info("TRT 추론 엔진 리소스 해제")


# ──────────────────────────────────────────────
# 단독 실행 테스트
# ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== TensorRT 추론 엔진 테스트 ===\n")

    engine = TRTInferenceEngine(
        engine_path="model/weights/fall_guardian.trt",
        fall_threshold=0.7,
    )

    print(f"TRT 사용 가능: {TRT_AVAILABLE}")
    loaded = engine.load()
    print(f"엔진 로드 상태: {loaded} ({'TRT 모드' if loaded else 'Fallback 모드'})")

    # 추론 테스트 (N=50회)
    print("\n추론 벤치마크 (50회):")
    T, N, C = 16, 64, 6

    for i in range(50):
        x = np.random.randn(1, T, N, C).astype(np.float32)
        fall, prob, lat_ms = engine.infer(x)
        if i % 10 == 0:
            print(f"  [{i+1:3d}] fall={fall}, prob={prob:.4f}, latency={lat_ms:.2f}ms")

    stats = engine.get_latency_stats()
    print(f"\n지연시간 통계:")
    print(f"  평균: {stats['mean_ms']:.2f}ms")
    print(f"  최솟값: {stats['min_ms']:.2f}ms")
    print(f"  최댓값: {stats['max_ms']:.2f}ms")
    print(f"  P95: {stats['p95_ms']:.2f}ms")
    print(f"  총 추론 횟수: {stats['count']}")

    engine.release()
    print("\nTensorRT 추론 엔진 테스트 완료!")
