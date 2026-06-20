"""
model/pointnet.py
=================
Time-distributed PointNet 구현.

PointNet(Qi et al., 2017) 아키텍처를 기반으로:
  - T-Net: 입력 포인트에 대한 affine 변환 행렬 학습 (3×3, 64×64)
  - PointNet: MLP(64,64) → MaxPool → MLP(128,1024) → MaxPool → 글로벌 특징 1024d
  - TimeDistributedPointNet: T 프레임 동시 배치 처리

입력: (Batch, T, N, input_dim)  — T: 시퀀스 길이, N: 포인트 수, input_dim: 특징 차원
출력: (Batch, T, 1024)          — 각 프레임의 글로벌 특징 벡터
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TNet(nn.Module):
    """
    Transformation Network (T-Net).

    입력 포인트 클라우드에 대한 k×k affine 변환 행렬을 학습한다.
    Orthogonality 정규화 손실(regularization)을 지원한다.

    Args:
        k: 변환 행렬 크기 (입력 T-Net: 3 또는 input_dim, 특징 T-Net: 64)
        input_dim: 입력 특징 차원 (첫 번째 T-Net에서 input_dim = k)
    """

    def __init__(self, k: int = 3, input_dim: int = 3) -> None:
        super().__init__()
        self.k = k

        # 포인트별 MLP (BatchNorm + ReLU 포함)
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # 글로벌 특징 → 변환 행렬 파라미터
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        # 항등 행렬로 초기화
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim, N) 형태의 입력

        Returns:
            trans: (B, k, k) 변환 행렬
        """
        batchsize = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))   # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))   # (B, 128, N)
        x = F.relu(self.bn3(self.conv3(x)))   # (B, 1024, N)

        # Global max pooling
        x = torch.max(x, 2, keepdim=True)[0]   # (B, 1024, 1)
        x = x.view(-1, 1024)                    # (B, 1024)

        x = F.relu(self.bn4(self.fc1(x)))       # (B, 512)
        x = F.relu(self.bn5(self.fc2(x)))       # (B, 256)
        x = self.fc3(x)                          # (B, k*k)

        # 항등 행렬 더하기
        iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1, self.k * self.k)
        iden = iden.repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)

        return x

    @staticmethod
    def orthogonality_loss(trans: torch.Tensor) -> torch.Tensor:
        """
        변환 행렬의 직교성 손실.
        L = ||I - A·A^T||_F^2

        Args:
            trans: (B, k, k) 변환 행렬

        Returns:
            스칼라 손실값
        """
        k = trans.size(1)
        I = torch.eye(k, dtype=trans.dtype, device=trans.device).unsqueeze(0)
        diff = I - torch.bmm(trans, trans.transpose(2, 1))
        return torch.mean(torch.norm(diff, dim=(1, 2)))


class PointNet(nn.Module):
    """
    단일 프레임 PointNet 특징 추출기.

    Architecture:
        입력 (B, N, input_dim)
        → T-Net(3×3) → 입력 변환
        → MLP(64, 64) + BN + ReLU (포인트별)
        → T-Net(64×64) → 특징 변환
        → MLP(128, 1024) + BN + ReLU
        → Global Max Pooling
        → 글로벌 특징 (B, 1024)

    Args:
        input_dim: 입력 포인트 특징 차원 (기본: 6 = x,y,z,vel,snr,noise)
        output_dim: 글로벌 특징 차원 (기본: 1024)
        use_tnet_input: 입력 T-Net 사용 여부
        use_tnet_feature: 특징 T-Net 사용 여부
    """

    def __init__(
        self,
        input_dim: int = 6,
        output_dim: int = 1024,
        use_tnet_input: bool = True,
        use_tnet_feature: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_tnet_input = use_tnet_input
        self.use_tnet_feature = use_tnet_feature

        # T-Net for input transformation
        if use_tnet_input:
            self.tnet_input = TNet(k=input_dim, input_dim=input_dim)

        # 첫 번째 MLP 블록: input_dim → 64 → 64
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        # T-Net for feature transformation
        if use_tnet_feature:
            self.tnet_feat = TNet(k=64, input_dim=64)

        # 두 번째 MLP 블록: 64 → 128 → output_dim
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, output_dim, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(output_dim)

        logger.debug(
            "PointNet 초기화: input_dim=%d, output_dim=%d",
            input_dim,
            output_dim,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x: (B, N, input_dim) 포인트 배열

        Returns:
            global_feat: (B, output_dim) 글로벌 특징
            trans_input: (B, input_dim, input_dim) 입력 변환 행렬 (또는 None)
            trans_feat: (B, 64, 64) 특징 변환 행렬 (또는 None)
        """
        B, N, C = x.shape
        trans_input = None
        trans_feat = None

        # (B, N, C) → (B, C, N) for Conv1d
        x = x.transpose(1, 2)  # (B, input_dim, N)

        # 입력 T-Net
        if self.use_tnet_input:
            trans_input = self.tnet_input(x)          # (B, C, C)
            x = x.transpose(2, 1)                     # (B, N, C)
            x = torch.bmm(x, trans_input)             # (B, N, C)
            x = x.transpose(2, 1)                     # (B, C, N)

        # 첫 번째 MLP 블록
        x = F.relu(self.bn1(self.conv1(x)))           # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))           # (B, 64, N)

        # 특징 T-Net
        if self.use_tnet_feature:
            trans_feat = self.tnet_feat(x)            # (B, 64, 64)
            x = x.transpose(2, 1)                     # (B, N, 64)
            x = torch.bmm(x, trans_feat)              # (B, N, 64)
            x = x.transpose(2, 1)                     # (B, 64, N)

        # 두 번째 MLP 블록
        x = F.relu(self.bn3(self.conv3(x)))           # (B, 128, N)
        x = F.relu(self.bn4(self.conv4(x)))           # (B, output_dim, N)

        # Global Max Pooling
        global_feat = torch.max(x, 2)[0]             # (B, output_dim)

        return global_feat, trans_input, trans_feat


class TimeDistributedPointNet(nn.Module):
    """
    T 프레임에 걸쳐 PointNet을 병렬 적용하는 Time-distributed 래퍼.

    입력: (Batch, T, N, input_dim)
    출력: (Batch, T, output_dim)

    배치 차원과 시간 차원을 합쳐서 PointNet을 한 번에 처리하므로 효율적이다.

    Args:
        input_dim: 포인트 특징 차원
        output_dim: 글로벌 특징 차원
        use_tnet_input: 입력 T-Net 사용 여부
        use_tnet_feature: 특징 T-Net 사용 여부
    """

    def __init__(
        self,
        input_dim: int = 6,
        output_dim: int = 1024,
        use_tnet_input: bool = True,
        use_tnet_feature: bool = True,
    ) -> None:
        super().__init__()
        self.pointnet = PointNet(
            input_dim=input_dim,
            output_dim=output_dim,
            use_tnet_input=use_tnet_input,
            use_tnet_feature=use_tnet_feature,
        )
        self.output_dim = output_dim
        logger.debug(
            "TimeDistributedPointNet 초기화: input_dim=%d, output_dim=%d",
            input_dim,
            output_dim,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x: (B, T, N, input_dim)

        Returns:
            out: (B, T, output_dim) 각 프레임 글로벌 특징
            trans_input: 마지막 프레임의 입력 변환 행렬
            trans_feat: 마지막 프레임의 특징 변환 행렬
        """
        B, T, N, C = x.shape

        # (B, T, N, C) → (B*T, N, C): 배치와 시간 차원 합치기
        x_flat = x.view(B * T, N, C)

        # PointNet 적용
        global_feat, trans_input, trans_feat = self.pointnet(x_flat)

        # (B*T, output_dim) → (B, T, output_dim)
        out = global_feat.view(B, T, self.output_dim)

        return out, trans_input, trans_feat


# ──────────────────────────────────────────────
# 단독 실행 테스트
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.DEBUG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== PointNet 테스트 (device: {device}) ===\n")

    # 단일 프레임 PointNet 테스트
    B, N, C = 4, 64, 6
    x = torch.randn(B, N, C, device=device)

    pn = PointNet(input_dim=C, output_dim=1024).to(device)
    t0 = time.perf_counter()
    feat, t_in, t_feat = pn(x)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[PointNet] 입력: {x.shape}")
    print(f"  출력 (글로벌 특징): {feat.shape}")
    print(f"  T-Net 입력 변환: {t_in.shape if t_in is not None else None}")
    print(f"  T-Net 특징 변환: {t_feat.shape if t_feat is not None else None}")
    print(f"  추론 시간: {elapsed:.2f}ms")

    # T-Net Orthogonality Loss 테스트
    orth_loss = TNet.orthogonality_loss(t_feat)
    print(f"  직교성 손실: {orth_loss.item():.4f}")

    # TimeDistributed PointNet 테스트
    print()
    B, T, N, C = 2, 16, 64, 6
    x_seq = torch.randn(B, T, N, C, device=device)

    td_pn = TimeDistributedPointNet(input_dim=C, output_dim=1024).to(device)
    t0 = time.perf_counter()
    out, _, _ = td_pn(x_seq)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[TimeDistributedPointNet] 입력: {x_seq.shape}")
    print(f"  출력: {out.shape}")
    print(f"  추론 시간: {elapsed:.2f}ms")

    # 파라미터 수
    total_params = sum(p.numel() for p in td_pn.parameters())
    print(f"  총 파라미터 수: {total_params:,}")
    print("\nPointNet 테스트 완료!")
