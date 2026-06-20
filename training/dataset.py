"""
training/dataset.py
===================
낙상/비낙상 포인트클라우드 시퀀스 데이터셋.

디렉토리 구조:
    data/
    ├── fall/
    │   ├── seq_001.npy    # (T, N, 6) 포인트 시퀀스
    │   └── ...
    └── normal/
        ├── seq_001.npy
        └── ...

각 .npy 파일은 전처리된 (T, N, input_dim) 배열이어야 한다.
raw .npy 파일을 지정하면 PointCloudPreprocessor로 온더플라이 처리 가능.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler

logger = logging.getLogger(__name__)


class FallDataset(Dataset):
    """
    낙상 감지 포인트클라우드 시퀀스 데이터셋.

    Args:
        data_root: 데이터 루트 디렉토리 (fall/, normal/ 서브폴더 포함)
        window_size: 시퀀스 프레임 수 (T)
        target_points: 프레임당 포인트 수 (N)
        input_dim: 포인트 특징 차원
        augment: 학습 시 데이터 증강 여부
        normalize: 정규화 여부
        cache_data: 데이터를 메모리에 캐싱 (빠른 학습용)
    """

    LABEL_MAP = {"fall": 1, "normal": 0}

    def __init__(
        self,
        data_root: str,
        window_size: int = 16,
        target_points: int = 64,
        input_dim: int = 6,
        augment: bool = False,
        normalize: bool = True,
        cache_data: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.window_size = window_size
        self.target_points = target_points
        self.input_dim = input_dim
        self.augment = augment
        self.normalize = normalize
        self.cache_data = cache_data

        self._samples: List[Tuple[Path, int]] = []  # [(파일경로, 레이블)]
        self._cache: Dict[int, np.ndarray] = {}

        self._load_file_list()
        if cache_data:
            self._load_all_to_cache()

        logger.info(
            "FallDataset 초기화: %d 샘플 (낙상=%d, 정상=%d)",
            len(self._samples),
            sum(1 for _, l in self._samples if l == 1),
            sum(1 for _, l in self._samples if l == 0),
        )

    def _load_file_list(self) -> None:
        """낙상/정상 파일 리스트 로드."""
        for cls_name, label in self.LABEL_MAP.items():
            cls_dir = self.data_root / cls_name
            if not cls_dir.exists():
                logger.warning("클래스 디렉토리 없음: %s", cls_dir)
                continue
            for fpath in sorted(cls_dir.glob("*.npy")):
                self._samples.append((fpath, label))

        if not self._samples:
            logger.warning(
                "데이터 파일이 없습니다. data_root=%s에 fall/, normal/ 폴더와 .npy 파일을 배치하세요.",
                self.data_root,
            )

    def _load_all_to_cache(self) -> None:
        """전체 데이터를 메모리에 로드."""
        logger.info("데이터 캐싱 시작 (%d 샘플)...", len(self._samples))
        for idx, (fpath, _) in enumerate(self._samples):
            self._cache[idx] = self._load_npy(fpath)
        logger.info("캐싱 완료")

    def _load_npy(self, fpath: Path) -> np.ndarray:
        """
        .npy 파일 로드 및 형태 검증.

        Returns:
            (T, N, input_dim) 배열
        """
        data = np.load(str(fpath)).astype(np.float32)

        # 형태 정규화
        if data.ndim == 2:
            # (N, input_dim) → 단일 프레임을 T번 반복
            data = np.tile(data[np.newaxis], (self.window_size, 1, 1))
        elif data.ndim == 3:
            T, N, C = data.shape
            # T 맞추기
            if T < self.window_size:
                pad = np.zeros((self.window_size - T, N, C), dtype=np.float32)
                data = np.vstack([data, pad])
            elif T > self.window_size:
                data = data[:self.window_size]
            # N 맞추기
            if N != self.target_points:
                data = self._resample_points(data)
        else:
            raise ValueError(f"지원하지 않는 데이터 형태: {data.shape}")

        return data

    def _resample_points(self, data: np.ndarray) -> np.ndarray:
        """포인트 수를 target_points로 재샘플링."""
        T, N, C = data.shape
        rng = np.random.default_rng()
        result = np.zeros((T, self.target_points, C), dtype=np.float32)
        for t in range(T):
            pts = data[t]
            if N >= self.target_points:
                idx = rng.choice(N, self.target_points, replace=False)
            else:
                idx = rng.choice(N, self.target_points, replace=True)
            result[t] = pts[idx]
        return result

    def _augment(self, data: np.ndarray) -> np.ndarray:
        """
        데이터 증강.

        - 랜덤 회전 (yaw 축)
        - 랜덤 스케일 (±10%)
        - 랜덤 포인트 드롭아웃 (10%)
        - 가우시안 지터 (σ=0.01)
        """
        rng = np.random.default_rng()

        # 1. 랜덤 yaw 회전 (수평면)
        theta = rng.uniform(-np.pi / 6, np.pi / 6)  # ±30도
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot = np.array([[cos_t, -sin_t, 0],
                        [sin_t,  cos_t, 0],
                        [0,      0,     1]], dtype=np.float32)
        data_aug = data.copy()
        data_aug[:, :, :3] = (data[:, :, :3] @ rot.T)

        # 2. 랜덤 스케일
        scale = rng.uniform(0.9, 1.1)
        data_aug[:, :, :3] *= scale

        # 3. 가우시안 지터
        jitter = rng.normal(0, 0.01, data_aug[:, :, :3].shape).astype(np.float32)
        data_aug[:, :, :3] += jitter

        # 4. 포인트 드롭아웃 (10%)
        if rng.random() < 0.5:
            n = data_aug.shape[1]
            drop_n = int(n * 0.1)
            keep_idx = rng.choice(n, n - drop_n, replace=False)
            pad_idx = rng.choice(keep_idx, drop_n, replace=True)
            all_idx = np.concatenate([keep_idx, pad_idx])
            data_aug = data_aug[:, all_idx, :]

        return data_aug.astype(np.float32)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            idx: 인덱스

        Returns:
            (data_tensor, label_tensor)
            data_tensor: (T, N, input_dim) float32
            label_tensor: 스칼라 long
        """
        fpath, label = self._samples[idx]

        if idx in self._cache:
            data = self._cache[idx].copy()
        else:
            data = self._load_npy(fpath)

        if self.augment and label == 1:
            # 낙상 클래스만 증강 (오버샘플 효과)
            if np.random.random() < 0.7:
                data = self._augment(data)

        return (
            torch.from_numpy(data),
            torch.tensor(label, dtype=torch.long),
        )

    def get_class_weights(self, indices: Optional[list] = None) -> torch.Tensor:
        """클래스 불균형 보정용 가중치 계산.

        Args:
            indices: 가중치를 계산할 대상 샘플 인덱스 목록. None이면 전체
                데이터셋 기준. random_split으로 얻은 Subset(train_ds)에 대해
                계산하려면 train_ds.indices를 전달해야 한다 (전체 데이터셋
                기준으로 계산하면 train/val/test 분할 비율이 반영되지 않는다).
        """
        samples = self._samples if indices is None else [self._samples[i] for i in indices]
        n_total = len(samples)
        n_fall = sum(1 for _, l in samples if l == 1)
        n_normal = n_total - n_fall

        if n_fall == 0 or n_normal == 0:
            return torch.ones(2)

        weight_normal = n_total / (2 * n_normal)
        weight_fall = n_total / (2 * n_fall)
        return torch.tensor([weight_normal, weight_fall], dtype=torch.float32)

    def get_weighted_sampler(self, indices: Optional[list] = None) -> WeightedRandomSampler:
        """불균형 클래스를 위한 WeightedRandomSampler 반환.

        Args:
            indices: 샘플러를 구성할 대상 샘플 인덱스 목록. random_split으로
                나뉜 train Subset에 사용할 때는 train_ds.indices를 전달해야
                한다. 전체 데이터셋 기준 가중치 목록을 그대로 Subset에 sampler로
                사용하면 Subset 길이보다 큰 인덱스가 뽑혀 IndexError가 발생한다.
        """
        samples = self._samples if indices is None else [self._samples[i] for i in indices]
        class_weights = self.get_class_weights(indices=indices)
        sample_weights = [class_weights[label].item() for _, label in samples]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    @staticmethod
    def generate_mock_dataset(
        output_dir: str,
        n_fall: int = 100,
        n_normal: int = 400,
        window_size: int = 16,
        target_points: int = 64,
        input_dim: int = 6,
        seed: int = 42,
    ) -> None:
        """
        테스트용 Mock 데이터셋 생성.

        Args:
            output_dir: 저장 디렉토리
            n_fall: 낙상 샘플 수
            n_normal: 정상 샘플 수
        """
        rng = np.random.default_rng(seed)
        root = Path(output_dir)

        for cls_name, n_samples, is_fall in [("fall", n_fall, True), ("normal", n_normal, False)]:
            cls_dir = root / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)

            for i in range(n_samples):
                T = window_size
                data = np.zeros((T, target_points, input_dim), dtype=np.float32)

                for t in range(T):
                    if is_fall:
                        # 낙상: z 낮고, 퍼진 형태
                        x = rng.normal(0, 0.5, target_points).astype(np.float32)
                        y = rng.normal(2.0, 0.5, target_points).astype(np.float32)
                        # 낙상 진행 시뮬레이션: 후반부로 갈수록 z 감소
                        z_center = max(0.1, 1.2 - t * 0.08)
                        z = rng.normal(z_center, 0.2, target_points).astype(np.float32)
                        vel = rng.normal(-0.5 - t * 0.05, 0.3, target_points).astype(np.float32)
                    else:
                        # 정상: z 높고 (서있음), 속도 거의 없음
                        x = rng.normal(0, 0.3, target_points).astype(np.float32)
                        y = rng.normal(2.0, 0.3, target_points).astype(np.float32)
                        z = rng.normal(1.0, 0.2, target_points).astype(np.float32)
                        vel = rng.normal(0.0, 0.1, target_points).astype(np.float32)

                    snr = rng.uniform(10, 40, target_points).astype(np.float32)
                    noise = rng.uniform(1, 5, target_points).astype(np.float32)
                    data[t] = np.column_stack([x, y, z, vel, snr, noise])

                np.save(str(cls_dir / f"seq_{i+1:04d}.npy"), data)

        logger.info(
            "Mock 데이터셋 생성 완료: %s (fall=%d, normal=%d)",
            output_dir, n_fall, n_normal,
        )


def create_dataloaders(
    data_root: str,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    학습/검증/테스트 DataLoader 생성.

    Args:
        data_root: 데이터 루트 경로
        batch_size: 배치 크기
        train_ratio: 학습 비율
        val_ratio: 검증 비율
        num_workers: 데이터 로더 워커 수
        use_weighted_sampler: 클래스 불균형 보정 sampler 사용 여부

    Returns:
        (train_loader, val_loader, test_loader)
    """
    dataset = FallDataset(data_root=data_root, augment=False, **dataset_kwargs)

    # 분할
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    # 학습 세트에 증강 적용
    train_ds.dataset.augment = True

    sampler = None
    shuffle_train = True
    if use_weighted_sampler and hasattr(train_ds.dataset, "get_weighted_sampler"):
        sampler = train_ds.dataset.get_weighted_sampler(indices=train_ds.indices)
        shuffle_train = False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        "DataLoader 생성: train=%d, val=%d, test=%d",
        len(train_ds), len(val_ds), len(test_ds),
    )
    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────
# 단독 실행 테스트
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    import os

    logging.basicConfig(level=logging.DEBUG)
    print("=== FallDataset 테스트 ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock 데이터 생성
        FallDataset.generate_mock_dataset(
            output_dir=tmpdir,
            n_fall=50,
            n_normal=200,
        )
        print(f"Mock 데이터 생성 위치: {tmpdir}")

        # 데이터셋 로드
        dataset = FallDataset(
            data_root=tmpdir,
            window_size=16,
            target_points=64,
            augment=True,
            cache_data=True,
        )

        print(f"전체 샘플 수: {len(dataset)}")
        print(f"클래스 가중치: {dataset.get_class_weights()}")

        # 샘플 확인
        data, label = dataset[0]
        print(f"\n첫 샘플:")
        print(f"  data shape: {data.shape}")
        print(f"  label: {label.item()} ({'낙상' if label.item() == 1 else '정상'})")
        print(f"  dtype: {data.dtype}")

        # DataLoader 테스트
        train_loader, val_loader, test_loader = create_dataloaders(
            data_root=tmpdir,
            batch_size=16,
            train_ratio=0.8,
            val_ratio=0.1,
            num_workers=0,  # 테스트환경
        )

        print(f"\nDataLoader 배치 테스트:")
        for batch_x, batch_y in train_loader:
            print(f"  배치 x shape: {batch_x.shape}")
            print(f"  배치 y shape: {batch_y.shape}")
            print(f"  낙상 샘플 수: {batch_y.sum().item()}")
            break

    print("\nFallDataset 테스트 완료!")
