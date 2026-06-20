"""
model/fall_detector.py
======================
통합 낙상 감지 모델 (FallDetector).

TimeDistributedPointNet + GRUClassifier를 하나의 엔드-투-엔드 모델로 통합.

전체 파이프라인:
    입력: (B, T, N, input_dim)
    → TimeDistributedPointNet → (B, T, 1024)
    → GRUClassifier → (B, 2)
    → Softmax
    출력: (B, 2) — [정상 확률, 낙상 확률]

학습 시 T-Net 직교성 정규화 손실을 추가하여 안정적인 수렴을 도모한다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.pointnet import TimeDistributedPointNet
from model.gru_classifier import GRUClassifier, FocalLoss

logger = logging.getLogger(__name__)


class FallDetector(nn.Module):
    """
    통합 낙상 감지 모델.

    Args:
        input_dim: 포인트 특징 차원 (기본 6: x,y,z,vel,snr,noise)
        pointnet_output_dim: PointNet 글로벌 특징 차원 (기본 1024)
        gru_hidden_size: GRU 히든 크기 (기본 256)
        gru_num_layers: GRU 레이어 수 (기본 2)
        gru_dropout: GRU 드롭아웃 (기본 0.3)
        num_classes: 분류 클래스 수 (기본 2)
        mlp_hidden: MLP 히든 크기 (기본 (128, 64))
        use_tnet_input: 입력 T-Net 사용 여부
        use_tnet_feature: 특징 T-Net 사용 여부
        fall_threshold: 낙상 판정 임계값 (기본 0.7)
    """

    def __init__(
        self,
        input_dim: int = 6,
        pointnet_output_dim: int = 1024,
        gru_hidden_size: int = 256,
        gru_num_layers: int = 2,
        gru_dropout: float = 0.3,
        num_classes: int = 2,
        mlp_hidden: Tuple[int, ...] = (128, 64),
        use_tnet_input: bool = True,
        use_tnet_feature: bool = True,
        fall_threshold: float = 0.7,
    ) -> None:
        super().__init__()
        self.fall_threshold = fall_threshold
        self.num_classes = num_classes

        # Time-distributed PointNet
        self.td_pointnet = TimeDistributedPointNet(
            input_dim=input_dim,
            output_dim=pointnet_output_dim,
            use_tnet_input=use_tnet_input,
            use_tnet_feature=use_tnet_feature,
        )

        # GRU 분류기
        self.gru_classifier = GRUClassifier(
            input_dim=pointnet_output_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            dropout=gru_dropout,
            num_classes=num_classes,
            mlp_hidden=mlp_hidden,
        )

        # 손실 함수
        self.criterion = FocalLoss(alpha=0.75, gamma=2.0)
        self.tnet_reg_weight = 0.001  # T-Net 직교성 손실 가중치

        logger.info(
            "FallDetector 초기화: input_dim=%d, threshold=%.2f",
            input_dim,
            fall_threshold,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, T, N, input_dim) 포인트클라우드 시퀀스

        Returns:
            logits: (B, num_classes)
            trans_feat: 특징 변환 행렬 (직교성 손실용)
        """
        # PointNet 특징 추출
        pn_features, trans_input, trans_feat = self.td_pointnet(x)
        # pn_features: (B, T, pointnet_output_dim)

        # GRU 분류
        logits, _ = self.gru_classifier(pn_features)
        # logits: (B, num_classes)

        return logits, trans_feat

    def compute_loss(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        학습 손실 계산.

        Args:
            x: (B, T, N, input_dim)
            targets: (B,) 정수 레이블 (0: 정상, 1: 낙상)

        Returns:
            total_loss: 스칼라 손실값
            loss_dict: {'focal_loss', 'tnet_loss', 'total_loss'}
        """
        logits, trans_feat = self.forward(x)

        # Focal Loss
        focal_loss = self.criterion(logits, targets)

        # T-Net 직교성 정규화
        from model.pointnet import TNet
        tnet_loss = torch.tensor(0.0, device=x.device)
        if trans_feat is not None:
            tnet_loss = TNet.orthogonality_loss(trans_feat)

        total_loss = focal_loss + self.tnet_reg_weight * tnet_loss

        return total_loss, {
            "focal_loss": focal_loss.item(),
            "tnet_loss": tnet_loss.item(),
            "total_loss": total_loss.item(),
        }

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        낙상 확률 반환.

        Args:
            x: (B, T, N, input_dim) 또는 (T, N, input_dim) 단일 샘플

        Returns:
            proba: (B, 2) 클래스 확률
        """
        was_unbatched = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            was_unbatched = True

        with torch.no_grad():
            logits, _ = self.forward(x)
            proba = F.softmax(logits, dim=-1)

        if was_unbatched:
            proba = proba.squeeze(0)

        return proba

    def predict(
        self,
        x: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """
        낙상 여부 단일 예측 (배치 크기 1).

        Args:
            x: (T, N, input_dim) 단일 시퀀스
            threshold: 낙상 판정 임계값 (None이면 self.fall_threshold 사용)

        Returns:
            (fall_detected: bool, fall_confidence: float)
        """
        if threshold is None:
            threshold = self.fall_threshold

        proba = self.predict_proba(x)
        if proba.dim() > 1:
            fall_prob = proba[0, 1].item()
        else:
            fall_prob = proba[1].item()

        return fall_prob >= threshold, fall_prob

    # ── 모델 저장/불러오기 ──────────────────────

    def save(self, path: str) -> None:
        """모델 가중치 저장."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "fall_threshold": self.fall_threshold,
                    "num_classes": self.num_classes,
                },
            },
            path,
        )
        logger.info("모델 저장: %s", path)

    def load(self, path: str, map_location: str = "cpu") -> None:
        """모델 가중치 불러오기."""
        checkpoint = torch.load(path, map_location=map_location)
        self.load_state_dict(checkpoint["state_dict"])
        if "config" in checkpoint:
            self.fall_threshold = checkpoint["config"].get(
                "fall_threshold", self.fall_threshold
            )
        logger.info("모델 불러오기: %s", path)

    def count_parameters(self) -> int:
        """학습 가능한 파라미터 수 반환."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────
# 단독 실행 테스트
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.DEBUG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== FallDetector 통합 모델 테스트 (device: {device}) ===\n")

    # 모델 생성
    model = FallDetector(
        input_dim=6,
        pointnet_output_dim=1024,
        gru_hidden_size=256,
        gru_num_layers=2,
        gru_dropout=0.3,
        fall_threshold=0.7,
    ).to(device)

    print(f"총 파라미터 수: {model.count_parameters():,}")

    # Forward pass 테스트
    B, T, N, C = 4, 16, 64, 6
    x = torch.randn(B, T, N, C, device=device)
    targets = torch.randint(0, 2, (B,), device=device)

    print(f"\n입력 shape: {x.shape}")

    # 학습 모드
    model.train()
    t0 = time.perf_counter()
    loss, loss_dict = model.compute_loss(x, targets)
    train_elapsed = (time.perf_counter() - t0) * 1000
    print(f"학습 손실: {loss.item():.4f}")
    print(f"손실 세부: {loss_dict}")
    print(f"학습 추론 시간: {train_elapsed:.2f}ms")

    # 추론 모드
    model.eval()
    t0 = time.perf_counter()
    proba = model.predict_proba(x)
    infer_elapsed = (time.perf_counter() - t0) * 1000
    print(f"\n예측 확률 shape: {proba.shape}")
    print(f"낙상 확률: {proba[:, 1].tolist()}")
    print(f"추론 시간: {infer_elapsed:.2f}ms")

    # 단일 샘플 예측
    x_single = torch.randn(T, N, C, device=device)
    fall_detected, confidence = model.predict(x_single)
    print(f"\n단일 샘플 예측: fall={fall_detected}, confidence={confidence:.4f}")

    # 저장/불러오기 테스트
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_model.pth")
        model.save(save_path)
        model.load(save_path)
        print(f"\n모델 저장/불러오기 테스트 완료")

    print("\nFallDetector 테스트 완료!")
