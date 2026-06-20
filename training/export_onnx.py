"""
training/export_onnx.py
=======================
학습된 FallDetector 모델을 ONNX 형식으로 변환.

ONNX 변환 후 model/tensorrt_inference.py의 TRTEngineBuilder로
TensorRT FP16 엔진으로 추가 변환 가능.

opset 기본값은 17이 아니라 13이다. 실제 배포 타깃인 Jetson Nano
(JetPack 4.6.x, TensorRT 8.2.1)의 내장 onnx-tensorrt 파서는 opset 13까지만
공식 지원하므로, 더 높은 opset으로 export하면 dev PC에서는 ONNX 변환/
onnxruntime 검증까지는 통과하더라도 실제 보드에서 TRT 엔진 빌드 시점에
실패한다.

사용법:
    python training/export_onnx.py \
        --checkpoint model/weights/best_checkpoint.pth \
        --output model/weights/fall_guardian.onnx \
        --opset 13
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from model.fall_detector import FallDetector

logger = logging.getLogger(__name__)


class FallDetectorONNX(nn.Module):
    """
    ONNX 변환을 위한 FallDetector 래퍼.

    TRT와 호환성을 위해:
    - dynamic axes 없이 고정 입력 크기 (1, 16, 64, 6)
    - 출력: (1, 2) softmax 확률
    - T-Net 직교성 손실 제거 (추론 전용)
    """

    def __init__(self, model: FallDetector) -> None:
        super().__init__()
        self.td_pointnet = model.td_pointnet
        self.gru_classifier = model.gru_classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, input_dim) float32

        Returns:
            proba: (B, 2) 클래스 확률
        """
        import torch.nn.functional as F
        pn_features, _, _ = self.td_pointnet(x)
        logits, _ = self.gru_classifier(pn_features)
        return F.softmax(logits, dim=-1)


def export_onnx(
    checkpoint_path: Optional[str],
    output_path: str,
    opset_version: int = 13,
    batch_size: int = 1,
    window_size: int = 16,
    num_points: int = 64,
    input_dim: int = 6,
    verify: bool = True,
) -> bool:
    """
    FallDetector를 ONNX로 변환 및 저장.

    Args:
        checkpoint_path: 체크포인트 경로 (None이면 랜덤 가중치)
        output_path: 저장할 ONNX 파일 경로
        opset_version: ONNX opset 버전. Jetson Nano(TRT 8.2.1)의 onnx-tensorrt
            파서가 opset 13까지만 공식 지원하므로 기본값을 13으로 둔다.
            더 높은 opset은 dev PC ONNX 변환에는 성공해도 실제 보드에서
            TRT 엔진 빌드가 실패할 수 있다.
        batch_size: 배치 크기 (TRT 최적화: 1)
        window_size: 윈도우 프레임 수
        num_points: 프레임당 포인트 수
        input_dim: 포인트 특징 차원
        verify: ONNX 검증 수행 여부

    Returns:
        성공 여부
    """
    logger.info("ONNX 변환 시작: %s", output_path)

    # 모델 로드
    model = FallDetector(input_dim=input_dim)
    if checkpoint_path and Path(checkpoint_path).exists():
        model.load(checkpoint_path, map_location="cpu")
        logger.info("체크포인트 로드: %s", checkpoint_path)
    else:
        logger.warning("체크포인트 없음. 랜덤 가중치 사용.")

    model.eval()
    onnx_model = FallDetectorONNX(model)
    onnx_model.eval()

    # 더미 입력
    dummy_input = torch.randn(batch_size, window_size, num_points, input_dim)
    logger.info("더미 입력 shape: %s", dummy_input.shape)

    # ONNX 변환
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            onnx_model,
            dummy_input,
            str(output_path_obj),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["point_cloud"],
            output_names=["fall_probability"],
            dynamic_axes={
                "point_cloud": {0: "batch_size"},
                "fall_probability": {0: "batch_size"},
            },
            verbose=False,
            # torch>=2.x는 dynamo=True(신규 ExportedProgram 기반 익스포터)가 기본값이며
            # onnxscript 패키지를 요구한다. Jetson Nano(TRT 8.2.1)용 opset<=13 호환성은
            # 레거시 TorchScript 기반 익스포터를 전제로 검증했으므로 명시적으로 비활성화한다.
            dynamo=False,
        )
        file_size_mb = output_path_obj.stat().st_size / (1024 * 1024)
        logger.info("ONNX 저장 완료: %s (%.1f MB)", output_path, file_size_mb)
    except Exception as e:
        logger.error("ONNX 변환 실패: %s", e)
        return False

    # ONNX 검증
    if verify:
        try:
            import onnx  # type: ignore
            onnx_model_check = onnx.load(str(output_path_obj))
            onnx.checker.check_model(onnx_model_check)
            logger.info("ONNX 모델 검증 완료")

            # ONNX Runtime으로 추론 테스트
            try:
                import onnxruntime as ort  # type: ignore
                sess = ort.InferenceSession(
                    str(output_path_obj),
                    providers=["CPUExecutionProvider"],
                )
                input_name = sess.get_inputs()[0].name
                result = sess.run(None, {input_name: dummy_input.numpy()})
                logger.info("ONNX Runtime 추론 테스트 완료: output shape=%s", result[0].shape)
            except ImportError:
                logger.info("onnxruntime 미설치. 추론 테스트 스킵")

        except ImportError:
            logger.info("onnx 패키지 미설치. 검증 스킵")

    return True


def export_trt(onnx_path: str, trt_path: str) -> bool:
    """
    ONNX → TensorRT FP16 변환.

    Args:
        onnx_path: ONNX 파일 경로
        trt_path: 저장할 TRT 엔진 경로

    Returns:
        성공 여부
    """
    try:
        from model.tensorrt_inference import TRTEngineBuilder, TRT_AVAILABLE
        if not TRT_AVAILABLE:
            logger.warning("TensorRT 미설치. TRT 변환 스킵")
            return False

        builder = TRTEngineBuilder(workspace_gb=4.0, fp16_mode=True)
        return builder.build_from_onnx(onnx_path, trt_path)
    except Exception as e:
        logger.error("TRT 변환 오류: %s", e)
        return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="FallDetector ONNX 변환")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="model/weights/fall_guardian.onnx")
    parser.add_argument("--opset", type=int, default=13, help="Jetson Nano TRT 8.2.1은 opset<=13만 공식 지원")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--export_trt", action="store_true", help="TRT 변환도 수행")
    args = parser.parse_args()

    print("=== FallDetector ONNX 변환 ===\n")
    success = export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
        batch_size=args.batch_size,
        verify=True,
    )

    if success:
        print(f"ONNX 변환 완료: {args.output}")
        if args.export_trt:
            trt_path = args.output.replace(".onnx", ".trt")
            trt_success = export_trt(args.output, trt_path)
            if trt_success:
                print(f"TRT 변환 완료: {trt_path}")
            else:
                print("TRT 변환 실패 (TensorRT 미설치 또는 오류)")
    else:
        print("ONNX 변환 실패!")
