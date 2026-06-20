"""
model/gru_classifier.py
=======================
GRU + MLP 낙상 분류기.

PointNet에서 추출한 시계열 특징 벡터를 GRU로 처리하고,
마지막 타임스텝의 hidden state를 MLP로 분류한다.

아키텍처:
    입력: (B, T, pointnet_dim)
    → GRU(hidden=256, layers=2, dropout=0.3)
    → 마지막 hidden state (B, 256)
    → MLP: Linear(256→128) → BN → ReLU → Dropout
    → Linear(128→64) → BN → ReLU → Dropout
    → Linear(64→2) → Softmax
    출력: (B, 2) — [정상 확률, 낙상 확률]
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GRUClassifier(nn.Module):
    """
    GRU 기반 시계열 낙상 분류기.

    Args:
        input_dim: 입력 특징 차원 (PointNet 출력, 기본 1024)
        hidden_size: GRU 히든 크기 (기본 256)
        num_layers: GRU 레이어 수 (기본 2)
        dropout: 드롭아웃 비율 (기본 0.3)
        num_classes: 출력 클래스 수 (기본 2: 정상/낙상)
        mlp_hidden: MLP 히든 레이어 크기 (기본 [128, 64])
        bidirectional: 양방향 GRU 사용 여부
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
        mlp_hidden: Tuple[int, ...] = (128, 64),
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # GRU 레이어
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # MLP 분류기
        gru_output_dim = hidden_size * self.num_directions
        mlp_layers = []
        prev_dim = gru_output_dim
        for h_dim in mlp_hidden:
            mlp_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ])
            prev_dim = h_dim

        mlp_layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*mlp_layers)

        # 가중치 초기화
        self._init_weights()

        logger.debug(
            "GRUClassifier 초기화: input=%d, hidden=%d, layers=%d, classes=%d",
            input_dim,
            hidden_size,
            num_layers,
            num_classes,
        )

    def _init_weights(self) -> None:
        """Orthogonal 초기화 (GRU 수렴 안정화)."""
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, input_dim) 시계열 특징
            h0: (num_layers * num_directions, B, hidden_size) 초기 hidden state

        Returns:
            logits: (B, num_classes) 분류 로짓
            last_hidden: (B, hidden_size * num_directions) 마지막 hidden state
        """
        B, T, _ = x.shape

        # GRU 처리
        out, hn = self.gru(x, h0)
        # out: (B, T, hidden_size * num_directions)
        # hn: (num_layers * num_directions, B, hidden_size)

        # 마지막 레이어의 hidden state 추출
        if self.bidirectional:
            # 양방향: 순방향과 역방향 마지막 hidden state 결합
            last_hidden = torch.cat([hn[-2], hn[-1]], dim=1)  # (B, hidden*2)
        else:
            last_hidden = hn[-1]  # (B, hidden_size)

        # MLP 분류
        logits = self.mlp(last_hidden)  # (B, num_classes)

        return logits, last_hidden

    def predict_proba(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Softmax 확률 반환.

        Args:
            x: (B, T, input_dim)

        Returns:
            proba: (B, num_classes) 클래스 확률
        """
        with torch.no_grad():
            logits, _ = self.forward(x, h0)
            return F.softmax(logits, dim=-1)

    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 0.7,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        낙상 여부 이진 예측.

        Args:
            x: (B, T, input_dim)
            threshold: 낙상 판정 임계값

        Returns:
            pred: (B,) 예측 클래스 (0: 정상, 1: 낙상)
            confidence: (B,) 낙상 확률
        """
        proba = self.predict_proba(x, h0)
        fall_prob = proba[:, 1]
        pred = (fall_prob >= threshold).long()
        return pred, fall_prob


class FocalLoss(nn.Module):
    """
    Focal Loss - 클래스 불균형 대응.

    낙상 데이터가 희소할 때 어려운 샘플에 더 집중.

    Args:
        alpha: 클래스 가중치 (낙상 클래스에 높은 가중치)
        gamma: 집중 파라미터 (gamma > 0이면 쉬운 샘플 하향)
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, num_classes) 로짓
            targets: (B,) 정수 레이블

        Returns:
            스칼라 손실값
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)

        # alpha 가중치 적용 (낙상 클래스 = 1)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ──────────────────────────────────────────────
# 단독 실행 테스트
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.DEBUG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== GRUClassifier 테스트 (device: {device}) ===\n")

    B, T, D = 8, 16, 1024

    model = GRUClassifier(
        input_dim=D,
        hidden_size=256,
        num_layers=2,
        dropout=0.3,
        num_classes=2,
        mlp_hidden=(128, 64),
    ).to(device)

    x = torch.randn(B, T, D, device=device)

    # Forward pass
    model.eval()
    t0 = time.perf_counter()
    logits, hidden = model(x)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"입력 shape: {x.shape}")
    print(f"로짓 shape: {logits.shape}")
    print(f"Hidden shape: {hidden.shape}")
    print(f"추론 시간: {elapsed:.2f}ms")

    # 확률 출력
    proba = model.predict_proba(x)
    print(f"확률 shape: {proba.shape}")
    print(f"낙상 확률 (첫 샘플): {proba[0, 1].item():.4f}")

    # 이진 예측
    pred, confidence = model.predict(x, threshold=0.7)
    print(f"예측: {pred.tolist()}")
    print(f"낙상 신뢰도: {[f'{c:.3f}' for c in confidence.tolist()]}")

    # Focal Loss 테스트
    print("\n[Focal Loss 테스트]")
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    model.train()
    logits_train, _ = model(x)
    targets = torch.randint(0, 2, (B,), device=device)
    loss = criterion(logits_train, targets)
    print(f"Focal Loss: {loss.item():.4f}")

    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n총 파라미터 수: {total_params:,}")
    print("\nGRUClassifier 테스트 완료!")
