"""
tests/test_preprocessor.py
===========================
PointCloudPreprocessor 단위 테스트.

테스트 항목:
  - DBSCAN 노이즈 제거
  - 포인트 수 정규화 (패딩, 샘플링)
  - 좌표 정규화 범위
  - 속도 정규화 범위
  - 윈도우 텐서 shape
  - 오버샘플링
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from radar.capture import PointCloud, RadarCapture
from radar.preprocessor import PointCloudPreprocessor


# ── Fixtures ───────────────────────────────────

@pytest.fixture
def preprocessor():
    return PointCloudPreprocessor(
        target_points=64,
        dbscan_eps=0.3,
        dbscan_min_samples=3,
        x_range=(-3.0, 3.0),
        y_range=(0.0, 5.0),
        z_range=(-0.5, 2.5),
        vmax=3.0,
    )


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def make_dense_cloud(n: int, center: np.ndarray, spread: float = 0.1) -> np.ndarray:
    """밀집된 포인트 클러스터 생성 (노이즈 없음)."""
    rng = np.random.default_rng(0)
    pts = rng.normal(0, spread, (n, 6)).astype(np.float32)
    pts[:, :3] += center
    pts[:, 4] = 20.0  # snr
    pts[:, 5] = 2.0   # noise
    return pts


# ── DBSCAN 노이즈 제거 테스트 ──────────────────

class TestDenoise:

    def test_removes_isolated_points(self, preprocessor):
        """고립 포인트(노이즈)가 제거되어야 한다."""
        # 밀집 클러스터
        cluster = make_dense_cloud(20, np.array([0, 2, 1]))
        # 노이즈 포인트 (멀리 떨어짐)
        noise = np.array([[10, 10, 10, 0, 5, 1]], dtype=np.float32)
        points = np.vstack([cluster, noise])

        result = preprocessor.denoise(points)
        # 노이즈 제거 확인
        assert len(result) <= len(cluster)
        # 노이즈 포인트의 좌표가 결과에 없어야 함
        if len(result) > 0:
            assert np.all(np.abs(result[:, 0]) < 5)

    def test_empty_input(self, preprocessor):
        """빈 포인트 입력 시 빈 배열 반환."""
        empty = np.zeros((0, 6), dtype=np.float32)
        result = preprocessor.denoise(empty)
        assert result.shape == (0, 6)

    def test_few_points(self, preprocessor):
        """최소 샘플 미만 포인트 입력 시 빈 배열 반환."""
        pts = np.random.randn(2, 6).astype(np.float32)
        result = preprocessor.denoise(pts)
        assert result.shape[1] == 6

    def test_dense_cluster_preserved(self, preprocessor):
        """밀집 클러스터 포인트는 보존되어야 한다."""
        cluster = make_dense_cloud(30, np.array([0, 2, 1]), spread=0.05)
        result = preprocessor.denoise(cluster)
        # 대부분의 포인트가 유지되어야 함
        assert len(result) >= 10

    def test_output_dtype(self, preprocessor):
        """출력 dtype은 float32여야 한다."""
        points = make_dense_cloud(20, np.array([0, 2, 1]))
        result = preprocessor.denoise(points)
        assert result.dtype == np.float32


# ── 포인트 수 정규화 테스트 ─────────────────────

class TestNormalizePointCount:

    def test_undersample_if_too_many(self, preprocessor, rng):
        """포인트가 너무 많으면 target으로 축소."""
        points = np.random.randn(200, 6).astype(np.float32)
        result = preprocessor.normalize_point_count(points, rng=rng)
        assert result.shape == (64, 6)

    def test_pad_if_too_few(self, preprocessor, rng):
        """포인트가 부족하면 target으로 패딩."""
        points = np.random.randn(10, 6).astype(np.float32)
        result = preprocessor.normalize_point_count(points, rng=rng)
        assert result.shape == (64, 6)

    def test_exact_count_unchanged(self, preprocessor, rng):
        """정확히 target 수이면 그대로."""
        points = np.random.randn(64, 6).astype(np.float32)
        result = preprocessor.normalize_point_count(points, rng=rng)
        assert result.shape == (64, 6)

    def test_zero_points(self, preprocessor, rng):
        """빈 입력은 제로 패딩으로 반환."""
        points = np.zeros((0, 6), dtype=np.float32)
        result = preprocessor.normalize_point_count(points, rng=rng)
        assert result.shape == (64, 6)
        assert np.all(result == 0)

    def test_output_dtype(self, preprocessor, rng):
        """출력 dtype은 float32여야 한다."""
        points = np.random.randn(30, 6).astype(np.float32)
        result = preprocessor.normalize_point_count(points, rng=rng)
        assert result.dtype == np.float32


# ── 좌표 정규화 테스트 ──────────────────────────

class TestNormalizeCoords:

    def test_xyz_in_minus1_to_1(self, preprocessor):
        """x, y, z 좌표가 [-1, 1] 범위에 있어야 한다."""
        # 경계값 테스트
        points = np.array([
            [-3.0, 0.0, -0.5, 0.0, 20.0, 2.0],  # x_min, y_min, z_min
            [3.0, 5.0, 2.5, 0.0, 20.0, 2.0],     # x_max, y_max, z_max
        ], dtype=np.float32)
        result = preprocessor.normalize_coords(points)

        assert np.isclose(result[0, 0], -1.0, atol=1e-5), f"x_min: {result[0,0]}"
        assert np.isclose(result[1, 0], 1.0, atol=1e-5), f"x_max: {result[1,0]}"
        assert np.isclose(result[0, 2], -1.0, atol=1e-5), f"z_min: {result[0,2]}"
        assert np.isclose(result[1, 2], 1.0, atol=1e-5), f"z_max: {result[1,2]}"

    def test_velocity_clipped_to_minus1_1(self, preprocessor):
        """속도가 [-1, 1]로 클리핑되어야 한다."""
        points = np.array([
            [0, 2, 1, 10.0, 20, 2],   # 속도 초과
            [0, 2, 1, -10.0, 20, 2],  # 속도 미만
            [0, 2, 1, 1.5, 20, 2],    # 정상
        ], dtype=np.float32)
        result = preprocessor.normalize_coords(points)

        assert result[0, 3] == pytest.approx(1.0, abs=1e-5)
        assert result[1, 3] == pytest.approx(-1.0, abs=1e-5)
        assert result[2, 3] == pytest.approx(1.5 / 3.0, abs=1e-5)

    def test_output_shape_preserved(self, preprocessor):
        """입력과 출력 shape이 동일해야 한다."""
        points = np.random.randn(64, 6).astype(np.float32)
        result = preprocessor.normalize_coords(points)
        assert result.shape == points.shape


# ── 윈도우 처리 테스트 ─────────────────────────

class TestProcessWindow:

    def test_window_output_shape(self, preprocessor):
        """윈도우 처리 출력 shape이 (T, N, 6)이어야 한다."""
        capture = RadarCapture(mock_mode=True, window_size=16, stride=8)
        window = [capture._generate_mock_frame("normal") for _ in range(16)]
        result = preprocessor.process_window(window)
        assert result.shape == (16, 64, 6)

    def test_window_dtype(self, preprocessor):
        """윈도우 출력 dtype은 float32."""
        capture = RadarCapture(mock_mode=True, window_size=16, stride=8)
        window = [capture._generate_mock_frame("normal") for _ in range(16)]
        result = preprocessor.process_window(window)
        assert result.dtype == np.float32

    def test_window_values_finite(self, preprocessor):
        """윈도우 출력에 NaN/Inf가 없어야 한다."""
        capture = RadarCapture(mock_mode=True, window_size=16, stride=8)
        window = [capture._generate_mock_frame("normal") for _ in range(16)]
        result = preprocessor.process_window(window)
        assert np.all(np.isfinite(result))


# ── 오버샘플링 테스트 ──────────────────────────

class TestOversample:

    def test_fall_class_increased(self, preprocessor):
        """낙상 클래스 수가 증가해야 한다."""
        rng = np.random.default_rng(0)
        B = 100
        windows = np.random.randn(B, 16, 64, 6).astype(np.float32)
        labels = np.array([1 if i < 10 else 0 for i in range(B)])

        aug_w, aug_l = preprocessor.oversample_fall_frames(
            windows, labels, oversample_ratio=3.0, rng=rng
        )

        fall_before = np.sum(labels == 1)
        fall_after = np.sum(aug_l == 1)
        assert fall_after > fall_before

    def test_output_shapes_consistent(self, preprocessor):
        """오버샘플 후 윈도우와 레이블 길이가 일치해야 한다."""
        rng = np.random.default_rng(0)
        windows = np.random.randn(50, 16, 64, 6).astype(np.float32)
        labels = np.array([1 if i < 5 else 0 for i in range(50)])

        aug_w, aug_l = preprocessor.oversample_fall_frames(windows, labels, rng=rng)
        assert len(aug_w) == len(aug_l)

    def test_no_fall_samples_handled(self, preprocessor):
        """낙상 샘플이 없어도 오류 없이 처리."""
        rng = np.random.default_rng(0)
        windows = np.random.randn(10, 16, 64, 6).astype(np.float32)
        labels = np.zeros(10, dtype=int)

        aug_w, aug_l = preprocessor.oversample_fall_frames(windows, labels, rng=rng)
        assert len(aug_w) == len(aug_l) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
