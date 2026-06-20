"""
radar/multi_tracker.py
======================
다중 인원 추적 모듈.

DBSCAN으로 레이더 포인트클라우드를 인원별 클러스터로 분리하고,
Kalman Filter로 각 인원의 궤적을 추적한다.
인원별 고유 ID를 유지하며 입장/퇴장 처리를 수행한다.
각 인원에 대해 독립적으로 낙상 감지를 적용할 수 있다.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN  # type: ignore

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 데이터 클래스
# ──────────────────────────────────────────────

@dataclass
class TrackedPerson:
    """
    추적 중인 개인의 상태.

    Attributes:
        person_id: 고유 ID
        position: 현재 추정 위치 (x, y, z)
        velocity_vec: 추정 속도 벡터 (vx, vy, vz)
        last_seen: 마지막으로 관측된 시각
        age_frames: 추적 유지 프레임 수
        points_history: 최근 프레임의 포인트들
        is_active: 현재 활성 상태
        fall_detected: 낙상 감지 여부
        fall_confidence: 낙상 확률
        fall_detected_time: 낙상 감지 시각
    """
    person_id: int
    position: np.ndarray                  # shape (3,): [x, y, z]
    velocity_vec: np.ndarray              # shape (3,)
    last_seen: float = field(default_factory=time.time)
    age_frames: int = 0
    points_history: List[np.ndarray] = field(default_factory=list)
    is_active: bool = True
    fall_detected: bool = False
    fall_confidence: float = 0.0
    fall_detected_time: Optional[float] = None

    def mark_fall(self, confidence: float) -> None:
        """낙상으로 표시."""
        self.fall_detected = True
        self.fall_confidence = confidence
        if self.fall_detected_time is None:
            self.fall_detected_time = time.time()

    def clear_fall(self) -> None:
        """낙상 해제."""
        self.fall_detected = False
        self.fall_confidence = 0.0
        self.fall_detected_time = None

    @property
    def fall_duration(self) -> float:
        """낙상 감지 후 경과 시간 (초)."""
        if self.fall_detected_time is None:
            return 0.0
        return time.time() - self.fall_detected_time


class KalmanFilter3D:
    """
    3D 위치 추적을 위한 단순 Kalman Filter.

    상태 벡터: [x, y, z, vx, vy, vz] (위치 + 속도)
    관측 벡터: [x, y, z] (위치만)
    """

    def __init__(
        self,
        process_noise: float = 0.1,
        measurement_noise: float = 0.5,
        initial_position: Optional[np.ndarray] = None,
    ) -> None:
        self.dt = 0.1  # 레이더 프레임 주기 (10 Hz)
        n_state = 6   # [x, y, z, vx, vy, vz]
        n_obs = 3     # [x, y, z]

        # 상태 전이 행렬 F
        self.F = np.eye(n_state, dtype=np.float64)
        self.F[0, 3] = self.dt
        self.F[1, 4] = self.dt
        self.F[2, 5] = self.dt

        # 관측 행렬 H
        self.H = np.zeros((n_obs, n_state), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # 프로세스 노이즈 공분산 Q
        self.Q = np.eye(n_state, dtype=np.float64) * process_noise

        # 측정 노이즈 공분산 R
        self.R = np.eye(n_obs, dtype=np.float64) * measurement_noise

        # 초기 상태
        self.x = np.zeros(n_state, dtype=np.float64)
        if initial_position is not None:
            self.x[:3] = initial_position

        # 초기 공분산
        self.P = np.eye(n_state, dtype=np.float64) * 1.0

    def predict(self) -> np.ndarray:
        """예측 단계."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3].copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        갱신 단계.

        Args:
            measurement: (3,) 관측 위치 [x, y, z]

        Returns:
            갱신된 위치 추정값
        """
        z = measurement.astype(np.float64)
        y = z - self.H @ self.x                          # 혁신
        S = self.H @ self.P @ self.H.T + self.R          # 혁신 공분산
        K = self.P @ self.H.T @ np.linalg.inv(S)         # Kalman Gain
        self.x = self.x + K @ y
        I = np.eye(len(self.x))
        self.P = (I - K @ self.H) @ self.P
        return self.x[:3].copy()

    @property
    def position(self) -> np.ndarray:
        """현재 위치 추정."""
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        """현재 속도 추정."""
        return self.x[3:].copy()


# ──────────────────────────────────────────────
# 다중 인원 추적기
# ──────────────────────────────────────────────

class MultiPersonTracker:
    """
    다중 인원 Kalman Filter 기반 추적기.

    레이더 포인트클라우드를 입력받아:
    1. DBSCAN으로 인원별 클러스터 분리
    2. 이전 프레임 추적 대상과 클러스터 매칭 (헝가리안 알고리즘 기반)
    3. Kalman Filter로 각 인원의 위치/속도 갱신
    4. 장시간 미관측 트랙 삭제

    Args:
        max_persons: 최대 추적 인원 수
        track_timeout: 트랙 유지 타임아웃 (초)
        min_track_frames: 최소 확정 프레임 수
        dbscan_eps: 인원 분리용 DBSCAN eps
        dbscan_min_samples: 인원 분리용 DBSCAN 최소 샘플
        process_noise: Kalman Filter 프로세스 노이즈
        measurement_noise: Kalman Filter 측정 노이즈
        association_distance: 트랙-관측 매칭 최대 거리 (m)
    """

    def __init__(
        self,
        max_persons: int = 5,
        track_timeout: float = 5.0,
        min_track_frames: int = 3,
        dbscan_eps: float = 0.6,
        dbscan_min_samples: int = 3,
        process_noise: float = 0.1,
        measurement_noise: float = 0.5,
        association_distance: float = 1.5,
    ) -> None:
        self.max_persons = max_persons
        self.track_timeout = track_timeout
        self.min_track_frames = min_track_frames
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.association_distance = association_distance

        self._tracks: Dict[int, TrackedPerson] = {}
        self._kalman_filters: Dict[int, KalmanFilter3D] = {}
        self._next_id: int = 1
        self._frame_count: int = 0

        self._dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)

        logger.info(
            "MultiPersonTracker 초기화: max_persons=%d, timeout=%.1fs",
            max_persons,
            track_timeout,
        )

    # ── 클러스터 분리 ───────────────────────────

    def _cluster_persons(
        self, points: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        DBSCAN으로 포인트를 인원별 클러스터로 분리.

        Args:
            points: (N, 6) 포인트 배열

        Returns:
            [(클러스터 포인트들, 클러스터 중심)] 리스트
        """
        if len(points) < self.dbscan_min_samples:
            return []

        xyz = points[:, :3]
        labels = self._dbscan.fit_predict(xyz)

        clusters = []
        unique_labels = set(labels) - {-1}

        for label in unique_labels:
            mask = labels == label
            cluster_pts = points[mask]
            centroid = cluster_pts[:, :3].mean(axis=0)
            clusters.append((cluster_pts, centroid))

        return clusters

    # ── 트랙-관측 매칭 ─────────────────────────

    def _associate_tracks(
        self,
        track_positions: Dict[int, np.ndarray],
        cluster_centroids: List[np.ndarray],
    ) -> Tuple[Dict[int, int], List[int], List[int]]:
        """
        Greedy 헝가리안 매칭으로 기존 트랙과 새 클러스터 매칭.

        Args:
            track_positions: {track_id: predicted_position}
            cluster_centroids: 새 클러스터 중심점 리스트

        Returns:
            (matched: {track_id: cluster_idx},
             unmatched_tracks: [track_id],
             unmatched_clusters: [cluster_idx])
        """
        if not track_positions or not cluster_centroids:
            return {}, list(track_positions.keys()), list(range(len(cluster_centroids)))

        track_ids = list(track_positions.keys())
        n_tracks = len(track_ids)
        n_clusters = len(cluster_centroids)

        # 비용 행렬 (거리)
        cost = np.full((n_tracks, n_clusters), fill_value=1e9)
        for i, tid in enumerate(track_ids):
            for j, centroid in enumerate(cluster_centroids):
                dist = np.linalg.norm(track_positions[tid] - centroid)
                cost[i, j] = dist

        matched = {}
        unmatched_tracks = set(track_ids)
        unmatched_clusters = set(range(n_clusters))

        # Greedy 매칭 (비용 최소부터)
        flat_sorted = np.argsort(cost.flatten())
        for idx in flat_sorted:
            i, j = divmod(int(idx), n_clusters)
            if cost[i, j] > self.association_distance:
                break
            tid = track_ids[i]
            if tid in unmatched_tracks and j in unmatched_clusters:
                matched[tid] = j
                unmatched_tracks.discard(tid)
                unmatched_clusters.discard(j)

        return matched, list(unmatched_tracks), list(unmatched_clusters)

    # ── 메인 업데이트 ───────────────────────────

    def update(self, points: np.ndarray) -> List[TrackedPerson]:
        """
        새 프레임 포인트클라우드로 추적기 갱신.

        Args:
            points: (N, 6) 포인트 배열 [x, y, z, vel, snr, noise]

        Returns:
            활성 TrackedPerson 리스트 (확정된 트랙만)
        """
        self._frame_count += 1
        now = time.time()

        # 1. 클러스터 분리
        clusters = self._cluster_persons(points)
        cluster_centroids = [c[1] for c in clusters]
        cluster_points = [c[0] for c in clusters]

        # 2. 기존 트랙 예측
        predicted_positions: Dict[int, np.ndarray] = {}
        for tid, kf in self._kalman_filters.items():
            predicted_positions[tid] = kf.predict()

        # 3. 매칭
        matched, unmatched_tracks, unmatched_clusters = self._associate_tracks(
            predicted_positions, cluster_centroids
        )

        # 4. 매칭된 트랙 업데이트
        for tid, cluster_idx in matched.items():
            centroid = cluster_centroids[cluster_idx]
            updated_pos = self._kalman_filters[tid].update(centroid)

            track = self._tracks[tid]
            track.position = updated_pos.astype(np.float32)
            track.velocity_vec = self._kalman_filters[tid].velocity.astype(np.float32)
            track.last_seen = now
            track.age_frames += 1
            track.points_history.append(cluster_points[cluster_idx])
            if len(track.points_history) > 16:
                track.points_history.pop(0)

        # 5. 매칭되지 않은 트랙 처리 (타임아웃)
        for tid in unmatched_tracks:
            track = self._tracks[tid]
            if now - track.last_seen > self.track_timeout:
                track.is_active = False
                logger.info("트랙 #%d 종료 (타임아웃)", tid)

        # 6. 새 클러스터로 신규 트랙 생성
        for cluster_idx in unmatched_clusters:
            if len(self._tracks) >= self.max_persons:
                logger.debug("최대 인원(%d) 초과, 신규 트랙 생성 스킵", self.max_persons)
                continue

            centroid = cluster_centroids[cluster_idx]
            new_id = self._next_id
            self._next_id += 1

            new_track = TrackedPerson(
                person_id=new_id,
                position=centroid.astype(np.float32),
                velocity_vec=np.zeros(3, dtype=np.float32),
            )
            new_track.points_history.append(cluster_points[cluster_idx])

            self._tracks[new_id] = new_track
            self._kalman_filters[new_id] = KalmanFilter3D(
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise,
                initial_position=centroid,
            )
            logger.debug("신규 트랙 #%d 생성: 위치=%s", new_id, centroid)

        # 7. 비활성 트랙 정리
        inactive_ids = [tid for tid, t in self._tracks.items() if not t.is_active]
        for tid in inactive_ids:
            del self._tracks[tid]
            del self._kalman_filters[tid]

        # 8. 확정된 활성 트랙만 반환
        active_tracks = [
            t for t in self._tracks.values()
            if t.is_active and t.age_frames >= self.min_track_frames
        ]

        logger.debug(
            "프레임 #%d: 클러스터=%d, 활성 트랙=%d",
            self._frame_count,
            len(clusters),
            len(active_tracks),
        )
        return active_tracks

    def update_fall_status(
        self,
        person_id: int,
        fall_detected: bool,
        confidence: float,
    ) -> None:
        """
        특정 인원의 낙상 상태 업데이트.

        Args:
            person_id: 대상 인원 ID
            fall_detected: 낙상 여부
            confidence: 낙상 확률
        """
        if person_id not in self._tracks:
            return

        track = self._tracks[person_id]
        if fall_detected:
            track.mark_fall(confidence)
            logger.warning(
                "트랙 #%d 낙상 감지 (confidence=%.3f)", person_id, confidence
            )
        else:
            if track.fall_detected:
                track.clear_fall()
                logger.info("트랙 #%d 낙상 해제", person_id)

    def get_person_window(
        self, person_id: int
    ) -> Optional[List[np.ndarray]]:
        """
        특정 인원의 최근 포인트 히스토리 반환.

        Args:
            person_id: 대상 인원 ID

        Returns:
            포인트 배열 리스트 또는 None
        """
        if person_id not in self._tracks:
            return None
        return self._tracks[person_id].points_history

    @property
    def active_persons(self) -> List[TrackedPerson]:
        """현재 활성 추적 인원 리스트."""
        return [t for t in self._tracks.values() if t.is_active]

    @property
    def fall_detected_persons(self) -> List[TrackedPerson]:
        """낙상이 감지된 인원 리스트."""
        return [t for t in self.active_persons if t.fall_detected]

    def reset(self) -> None:
        """추적기 초기화."""
        self._tracks.clear()
        self._kalman_filters.clear()
        self._next_id = 1
        self._frame_count = 0
        logger.info("MultiPersonTracker 초기화 완료")


# ──────────────────────────────────────────────
# 단독 실행 테스트
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import time

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=== MultiPersonTracker 테스트 ===\n")

    tracker = MultiPersonTracker(
        max_persons=5,
        track_timeout=2.0,
        min_track_frames=2,
    )

    rng = np.random.default_rng(42)

    # 2명의 가상 인물 시뮬레이션
    person_positions = [
        np.array([0.0, 2.0, 1.0]),   # 인물 A
        np.array([1.5, 3.0, 1.0]),   # 인물 B
    ]

    for frame_idx in range(20):
        all_points = []
        for pos in person_positions:
            # 각 인물 주변에 30개 포인트 생성
            pts = rng.normal(0, 0.15, (30, 6)).astype(np.float32)
            pts[:, :3] += pos
            all_points.append(pts)

        # 인물 B는 10 프레임 이후 낙상 (z 급락)
        if frame_idx >= 10:
            person_positions[1][2] = 0.1

        points = np.vstack(all_points)
        active_tracks = tracker.update(points)

        print(f"프레임 {frame_idx+1:02d}: 활성 트랙 {len(active_tracks)}명")
        for t in active_tracks:
            print(
                f"  ID={t.person_id} 위치=({t.position[0]:.2f}, "
                f"{t.position[1]:.2f}, {t.position[2]:.2f}) "
                f"age={t.age_frames}"
            )

        # 낙상 상태 시뮬레이션
        for t in active_tracks:
            if t.position[2] < 0.3 and frame_idx >= 10:
                tracker.update_fall_status(t.person_id, True, 0.95)

        time.sleep(0.05)

    print(f"\n낙상 감지 인원: {len(tracker.fall_detected_persons)}명")
    for t in tracker.fall_detected_persons:
        print(f"  ID={t.person_id}, 낙상 지속 시간={t.fall_duration:.1f}초")
