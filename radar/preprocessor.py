"""
radar/preprocessor.py
=====================
포인트클라우드 전처리 모듈.

주요 기능:
  1. DBSCAN으로 노이즈 및 바닥 반사 제거
  2. 포인트 수 정규화 (패딩 or 랜덤 샘플링으로 N=64)
  3. 좌표 및 속도 정규화 ([-1, 1] 스케일)
  4. 소수 클래스(낙상 프레임) 오버샘플링 (랜덤 지터 추가)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN  # type: ignore

from radar.capture import PointCloud

logger = logging.getLogger(__name__)


class PointCloudPreprocessor:
    """
    레이더 포인트클라우드 전처리기.

    파이프라인:
        raw PointCloud
        → DBSCAN 노이즈 제거
        → 유효 클러스터 필터링
        → N포인트로 정규화
        → 좌표/속도 정규화

    Args:
        target_points: 정규화할 포인트 수 (default: 64)
        dbscan_eps: DBSCAN epsilon 파라미터
        dbscan_min_samples: DBSCAN 최소 샘플 수
        x_range: x축 정규화 범위 [min, max] (m)
        y_range: y축 정규화 범위 [min, max] (m)
        z_range: z축 정규화 범위 [min, max] (m)
        vmax: 속도 최대값 (m/s)
        jitter_std: 오버샘플링 지터 표준편차
    """

    def __init__(
        self,
        target_points: int = 64,
        dbscan_eps: float = 0.3,
        dbscan_min_samples: int = 3,
        x_range: Tuple[float, float] = (-3.0, 3.0),
        y_range: Tuple[float, float] = (0.0, 5.0),
        z_range: Tuple[float, float] = (-0.5, 2.5),
        vmax: float = 3.0,
        jitter_std: float = 0.02,
    ) -> None:
        self.target_points = target_points
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.vmax = vmax
        self.jitter_std = jitter_std

        self._dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)

        logger.info(
            "PointCloudPreprocessor 초기화: target_points=%d, eps=%.2f",
            target_points,
            dbscan_eps,
        )

    # ── 개별 프레임 처리 ────────────────────────

    def denoise(self, points: np.ndarray) -> np.ndarray:
        """
        DBSCAN으로 노이즈 포인트 제거.

        레이블 -1(노이즈)로 분류된 포인트를 제거하고,
        유효 클러스터 포인트만 반환한다.

        Args:
            points: (N, 6) 포인트 배열 [x, y, z, vel, snr, noise]

        Returns:
            노이즈 제거된 (M, 6) 배열
        """
        if len(points) < self.dbscan_min_samples:
            logger.debug("포인트 수 부족(%d), 빈 배열 반환", len(points))
            return np.zeros((0, 6), dtype=np.float32)

        # XYZ 좌표만 사용하여 클러스터링
        xyz = points[:, :3]
        labels = self._dbscan.fit_predict(xyz)

        # 노이즈(-1) 제거
        mask = labels >= 0
        denoised = points[mask]
        removed = len(points) - len(denoised)

        if removed > 0:
            logger.debug("DBSCAN: %d/%d 포인트 노이즈 제거됨", removed, len(points))

        return denoised.astype(np.float32)

    def normalize_point_count(
        self,
        points: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        포인트 수를 target_points로 정규화.

        - 포인트 수 > target_points: 랜덤 샘플링으로 축소
        - 포인트 수 == 0: 제로 패딩
        - 포인트 수 < target_points: 기존 포인트를 반복 샘플링하여 패딩

        Args:
            points: (N, 6) 포인트 배열
            rng: 난수 생성기 (재현성 테스트용)

        Returns:
            (target_points, 6) 배열
        """
        if rng is None:
            rng = np.random.default_rng()

        n = len(points)

        if n == 0:
            return np.zeros((self.target_points, 6), dtype=np.float32)

        if n > self.target_points:
            idx = rng.choice(n, self.target_points, replace=False)
            return points[idx].astype(np.float32)

        if n < self.target_points:
            # 부족한 만큼 중복 샘플링 + 소량 지터
            deficit = self.target_points - n
            idx = rng.choice(n, deficit, replace=True)
            extras = points[idx] + rng.normal(0, self.jitter_std * 0.1, (deficit, 6)).astype(np.float32)
            return np.vstack([points, extras]).astype(np.float32)

        return points.astype(np.float32)

    def normalize_coords(self, points: np.ndarray) -> np.ndarray:
        """
        좌표 및 속도를 [-1, 1] 범위로 정규화.

        - x: x_range 기준
        - y: y_range 기준
        - z: z_range 기준
        - velocity: [-vmax, vmax] 기준
        - snr, noise: 변환 없음 (0-40dB 범위 유지)

        Args:
            points: (N, 6) 포인트 배열

        Returns:
            (N, 6) 정규화된 배열
        """
        normalized = points.copy()

        # x 정규화
        xmin, xmax = self.x_range
        normalized[:, 0] = 2.0 * (points[:, 0] - xmin) / (xmax - xmin) - 1.0

        # y 정규화
        ymin, ymax = self.y_range
        normalized[:, 1] = 2.0 * (points[:, 1] - ymin) / (ymax - ymin) - 1.0

        # z 정규화
        zmin, zmax = self.z_range
        normalized[:, 2] = 2.0 * (points[:, 2] - zmin) / (zmax - zmin) - 1.0

        # 속도 정규화
        normalized[:, 3] = np.clip(points[:, 3] / self.vmax, -1.0, 1.0)

        # snr, noise: 0~40 범위를 [0, 1]로 정규화
        normalized[:, 4] = np.clip(points[:, 4] / 40.0, 0.0, 1.0)
        normalized[:, 5] = np.clip(points[:, 5] / 10.0, 0.0, 1.0)

        return normalized.astype(np.float32)

    def process_frame(
        self,
        pc: PointCloud,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        단일 프레임 전처리 파이프라인.

        Args:
            pc: 입력 PointCloud
            rng: 난수 생성기

        Returns:
            (target_points, 6) 정규화된 포인트 배열
        """
        # 1. DBSCAN 노이즈 제거
        denoised = self.denoise(pc.points)

        # 2. 포인트 수 정규화
        normalized_count = self.normalize_point_count(denoised, rng=rng)

        # 3. 좌표 정규화
        normalized = self.normalize_coords(normalized_count)

        return normalized

    def process_window(
        self,
        window: List[PointCloud],
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        슬라이딩 윈도우(T 프레임) 전처리.

        Args:
            window: T 개의 PointCloud 리스트
            rng: 난수 생성기

        Returns:
            (T, target_points, 6) 텐서
        """
        if rng is None:
            rng = np.random.default_rng()

        frames = [self.process_frame(pc, rng=rng) for pc in window]
        tensor = np.stack(frames, axis=0)  # (T, N, 6)

        logger.debug("윈도우 처리 완료: shape=%s", tensor.shape)
        return tensor.astype(np.float32)

    # ── 오버샘플링 (학습용) ─────────────────────

    def oversample_fall_frames(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        oversample_ratio: float = 3.0,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        학습 데이터에서 낙상 클래스(label=1)를 오버샘플링.

        기존 낙상 샘플에 랜덤 지터를 추가하여 복사본 생성.

        Args:
            windows: (B, T, N, 6) 형태의 윈도우 배열
            labels: (B,) 라벨 배열 (0: 정상, 1: 낙상)
            oversample_ratio: 낙상 샘플을 몇 배로 늘릴지
            rng: 난수 생성기

        Returns:
            (오버샘플된 windows, 오버샘플된 labels)
        """
        if rng is None:
            rng = np.random.default_rng()

        fall_idx = np.where(labels == 1)[0]
        normal_idx = np.where(labels == 0)[0]

        if len(fall_idx) == 0:
            logger.warning("낙상 샘플이 없어 오버샘플링 스킵")
            return windows, labels

        logger.info(
            "오버샘플링 전: 정상=%d, 낙상=%d (비율=1:%.2f)",
            len(normal_idx),
            len(fall_idx),
            len(normal_idx) / max(len(fall_idx), 1),
        )

        # 오버샘플링할 개수
        target_fall = int(len(fall_idx) * oversample_ratio)
        sample_idx = rng.choice(fall_idx, target_fall, replace=True)

        # 지터 추가
        jitter = rng.normal(0, self.jitter_std, (target_fall, *windows.shape[1:])).astype(np.float32)
        oversampled_windows = windows[sample_idx] + jitter
        oversampled_labels = np.ones(target_fall, dtype=labels.dtype)

        # 병합
        all_windows = np.concatenate([windows, oversampled_windows], axis=0)
        all_labels = np.concatenate([labels, oversampled_labels], axis=0)

        # 셔플
        perm = rng.permutation(len(all_labels))
        all_windows = all_windows[perm]
        all_labels = all_labels[perm]

        logger.info(
            "오버샘플링 후: 정상=%d, 낙상=%d",
            np.sum(all_labels == 0),
            np.sum(all_labels == 1),
        )

        return all_windows.astype(np.float32), all_labels

    def compute_statistics(self, windows: np.ndarray) -> dict:
        """
        데이터셋 통계 계산 (정규화 파라미터 추정용).

        Args:
            windows: (B, T, N, 6) 배열

        Returns:
            평균, 표준편차, 최솟값, 최댓값 딕셔너리
        """
        flat = windows.reshape(-1, 6)
        return {
            "mean": flat.mean(axis=0).tolist(),
            "std": flat.std(axis=0).tolist(),
            "min": flat.min(axis=0).tolist(),
            "max": flat.max(axis=0).tolist(),
        }


# ──────────────────────────────────────────────
# 단독 실행 테스트
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.DEBUG)

    from radar.capture import RadarCapture

    print("=== PointCloudPreprocessor 테스트 ===\n")

    preprocessor = PointCloudPreprocessor(
        target_points=64,
        dbscan_eps=0.3,
        dbscan_min_samples=3,
    )
    capture = RadarCapture(mock_mode=True, window_size=16, stride=8)

    # 단일 프레임 테스트
    pc_normal = capture._generate_mock_frame("normal")
    pc_fall = capture._generate_mock_frame("fall")

    print(f"[정상 프레임] 원본 포인트 수: {pc_normal.num_points}")
    result_normal = preprocessor.process_frame(pc_normal)
    print(f"  전처리 후 shape: {result_normal.shape}")
    print(f"  x 범위: [{result_normal[:, 0].min():.3f}, {result_normal[:, 0].max():.3f}]")
    print(f"  z 범위: [{result_normal[:, 2].min():.3f}, {result_normal[:, 2].max():.3f}]")

    print(f"\n[낙상 프레임] 원본 포인트 수: {pc_fall.num_points}")
    result_fall = preprocessor.process_frame(pc_fall)
    print(f"  전처리 후 shape: {result_fall.shape}")
    print(f"  z 범위: [{result_fall[:, 2].min():.3f}, {result_fall[:, 2].max():.3f}]")

    # 윈도우 테스트
    print("\n[윈도우 처리 테스트]")
    window = [capture._generate_mock_frame("normal") for _ in range(16)]
    t0 = time.perf_counter()
    tensor = preprocessor.process_window(window)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"  윈도우 shape: {tensor.shape}")
    print(f"  처리 시간: {elapsed_ms:.2f}ms")
    print(f"  dtype: {tensor.dtype}")

    # 오버샘플링 테스트
    print("\n[오버샘플링 테스트]")
    B = 100
    dummy_windows = np.random.randn(B, 16, 64, 6).astype(np.float32)
    dummy_labels = np.array([1 if i < 10 else 0 for i in range(B)])
    aug_w, aug_l = preprocessor.oversample_fall_frames(dummy_windows, dummy_labels)
    print(f"  원본: {B}샘플 (낙상 10개)")
    print(f"  증강 후: {len(aug_l)}샘플 (낙상 {np.sum(aug_l==1)}개)")
    print("\n전처리 테스트 완료!")
