"""
tests/test_tensorrt_inference.py
==================================
TRTInferenceEngine / TRTEngineBuilder 단위 테스트.

개발 PC(샌드박스)에는 TensorRT/pycuda가 없으므로 (TRT_AVAILABLE=False)
실제 GPU 추론 경로는 검증할 수 없다. 대신 실제 Jetson Nano 없이도
검증 가능한 두 가지를 테스트한다:
  - TRT 미사용 시 정상적으로 폴백되는지 (엔진 빌드/로드 실패 처리)
  - ONNX Runtime → 랜덤 순의 CPU 폴백 추론 경로 (model/fall_detector.py를
    실제로 ONNX로 export하여 onnxruntime 폴백까지 엔드투엔드로 확인)

테스트 항목:
  - TRT_AVAILABLE 플래그와 환경 일치
  - TRTEngineBuilder.build_from_onnx: TRT 미설치 시 False 반환
  - TRTInferenceEngine.load(): TRT 미설치 시 False 반환 (파일 유무 무관)
  - infer(): onnx 파일 없을 때 랜덤 폴백 (반환 타입/범위, 지연시간 기록)
  - infer(): onnx 파일 있을 때 onnxruntime 폴백 (실제 FallDetector 모델 사용)
  - get_latency_stats(): 빈 상태 / 누적 통계
  - release(): 예외 없이 리소스 정리
  - _softmax: 합이 1, 단조성
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from model.tensorrt_inference import TRT_AVAILABLE, TRTEngineBuilder, TRTInferenceEngine

T, N, C = 16, 64, 6


@pytest.fixture
def engine(tmp_path):
    return TRTInferenceEngine(
        engine_path=str(tmp_path / "fall_guardian.trt"),
        fall_threshold=0.7,
    )


@pytest.fixture(scope="module")
def exported_onnx_path(tmp_path_factory):
    """model/fall_detector.py의 FallDetector를 실제로 ONNX로 export."""
    torch = pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    from model.fall_detector import FallDetector

    out_dir = tmp_path_factory.mktemp("trt_onnx")
    onnx_path = out_dir / "fall_guardian.onnx"

    model = FallDetector()
    model.eval()
    dummy = torch.randn(1, T, N, C)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=13,
        dynamo=False,
    )
    return onnx_path


class TestTRTAvailability:
    def test_trt_not_available_in_sandbox(self):
        # 샌드박스에는 TensorRT/pycuda가 설치되어 있지 않으므로 항상 False.
        # 실제 Jetson Nano(JetPack 4.6.x, TRT 8.2.1)에서는 True여야 한다.
        assert TRT_AVAILABLE is False


class TestTRTEngineBuilderWithoutTRT:
    def test_build_from_onnx_returns_false_without_trt(self, tmp_path):
        builder = TRTEngineBuilder()
        ok = builder.build_from_onnx(
            onnx_path=str(tmp_path / "model.onnx"),
            engine_path=str(tmp_path / "model.trt"),
        )
        assert ok is False


class TestTRTInferenceEngineLoad:
    def test_load_returns_false_without_trt(self, engine):
        assert engine.load() is False
        assert engine._loaded is False

    def test_load_returns_false_even_if_engine_file_exists(self, tmp_path):
        fake_engine = tmp_path / "fall_guardian.trt"
        fake_engine.write_bytes(b"not a real trt engine")
        engine = TRTInferenceEngine(engine_path=str(fake_engine))
        # TRT_AVAILABLE=False 이므로 파일이 있어도 항상 폴백으로 빠진다.
        assert engine.load() is False


class TestRandomFallbackInference:
    def test_infer_without_onnx_file_uses_random_fallback(self, engine):
        x = np.random.randn(1, T, N, C).astype(np.float32)
        fall_detected, prob, latency_ms = engine.infer(x)
        assert isinstance(fall_detected, (bool, np.bool_))
        assert 0.0 <= prob <= 1.0
        assert latency_ms > 0.0

    def test_infer_accepts_3d_input_without_batch_dim(self, engine):
        x = np.random.randn(T, N, C).astype(np.float32)
        fall_detected, prob, latency_ms = engine.infer(x)
        assert 0.0 <= prob <= 1.0

    def test_random_fallback_records_latency_history(self, engine):
        assert engine.get_latency_stats()["count"] == 0
        for _ in range(5):
            engine.infer(np.random.randn(1, T, N, C).astype(np.float32))
        stats = engine.get_latency_stats()
        assert stats["count"] == 5
        assert stats["mean_ms"] > 0.0


class TestOnnxRuntimeFallbackInference:
    def test_infer_uses_onnxruntime_when_onnx_file_present(self, exported_onnx_path, tmp_path):
        pytest.importorskip("onnxruntime")

        # engine_path.replace(".trt", ".onnx") 가 exported_onnx_path를 가리키도록
        # 같은 디렉토리/파일명으로 엔진 경로를 구성한다.
        engine_path = tmp_path / "fall_guardian.trt"
        onnx_path = tmp_path / "fall_guardian.onnx"
        onnx_path.write_bytes(exported_onnx_path.read_bytes())

        engine = TRTInferenceEngine(engine_path=str(engine_path), fall_threshold=0.7)
        x = np.random.randn(1, T, N, C).astype(np.float32)
        fall_detected, prob, latency_ms = engine.infer(x)

        assert 0.0 <= prob <= 1.0
        assert fall_detected == (prob >= 0.7)
        assert latency_ms > 0.0
        assert engine.get_latency_stats()["count"] == 1


class TestLatencyStats:
    def test_empty_stats_before_any_inference(self, engine):
        stats = engine.get_latency_stats()
        assert stats == {"mean_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "p95_ms": 0.0, "count": 0}

    def test_stats_keys_present_after_inference(self, engine):
        engine.infer(np.random.randn(1, T, N, C).astype(np.float32))
        stats = engine.get_latency_stats()
        assert set(stats.keys()) == {"mean_ms", "min_ms", "max_ms", "p95_ms", "count"}


class TestRelease:
    def test_release_clears_buffers_without_error(self, engine):
        engine._inputs = [{"host": None, "device": None}]
        engine._outputs = [{"host": None, "device": None}]
        engine._bindings = [123]
        engine.release()
        assert engine._inputs == []
        assert engine._outputs == []
        assert engine._bindings == []

    def test_release_safe_when_never_loaded(self, engine):
        engine.release()  # 예외 없이 통과해야 함


class TestSoftmax:
    def test_softmax_sums_to_one(self, engine):
        x = np.array([1.0, 2.0, 3.0])
        proba = engine._softmax(x)
        assert proba.sum() == pytest.approx(1.0)

    def test_softmax_preserves_order(self, engine):
        x = np.array([0.1, 5.0, -3.0])
        proba = engine._softmax(x)
        assert proba[1] > proba[0] > proba[2]

    def test_softmax_numerically_stable_for_large_values(self, engine):
        x = np.array([1000.0, 1001.0])
        proba = engine._softmax(x)
        assert np.all(np.isfinite(proba))
        assert proba.sum() == pytest.approx(1.0)
