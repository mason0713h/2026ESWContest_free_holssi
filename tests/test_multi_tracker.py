"""
tests/test_multi_tracker.py
=============================
MultiPersonTracker 단위 테스트.

테스트 항목:
  - 신규 트랙 생성 및 min_track_frames 확정
  - Kalman Filter 기반 위치 추적 (이동하는 클러스터 추종)
  - 다중 인원 분리 추적 (2인 이상)
  - 트랙 타임아웃 처리 (장기 미관측 트랙 제거)
  - 낙상 상태 연속 프레임 확정 (오경보 억제) / 해제
  - get_person_window 히스토리 길이 제한
  - reset() 동작
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from radar.multi_tracker import MultiPersonTracker


def make_person_points(center, n=30, spread=0.1, rng=None):
    """center 주변에 흩어진 가상 포인트 (N, 6) 생성."""
    if rng is None:
        rng = np.random.default_rng(0)
    pts = rng.normal(0, spread, (n, 6)).astype(np.float32)
    pts[:, :3] += np.asarray(center, dtype=np.float32)
    return pts


@pytest.fixture
def tracker():
    return MultiPersonTracker(
        max_persons=5,
        track_timeout=0.2,
        min_track_frames=3,
        dbscan_eps=0.6,
        dbscan_min_samples=3,
        consecutive_frames=3,
    )


class TestTrackCreationAndConfirmation:
    def test_new_track_not_active_before_min_frames(self, tracker):
        pts = make_person_points([0.0, 2.0, 1.0])
        active = tracker.update(pts)
        # 첫 프레임은 min_track_frames(3) 미달이라 활성 트랙으로 반환되지 않음
        assert active == []
        assert len(tracker._tracks) == 1

    def test_track_confirmed_after_min_frames(self, tracker):
        pts = make_person_points([0.0, 2.0, 1.0])
        for _ in range(4):
            active = tracker.update(pts)
        assert len(active) == 1
        assert active[0].age_frames == 3

    def test_track_id_persists_across_frames(self, tracker):
        pts = make_person_points([0.0, 2.0, 1.0])
        ids = set()
        for _ in range(5):
            active = tracker.update(pts)
            ids.update(t.person_id for t in active)
        assert len(ids) == 1  # 같은 인원은 동일 ID 유지


class TestMultiPersonSeparation:
    def test_two_separated_persons_get_distinct_ids(self, tracker):
        p1 = [0.0, 2.0, 1.0]
        p2 = [3.0, 4.0, 1.0]  # association_distance(1.5)보다 충분히 멀리 분리
        for _ in range(4):
            pts = np.vstack([make_person_points(p1), make_person_points(p2)])
            active = tracker.update(pts)
        assert len(active) == 2
        assert active[0].person_id != active[1].person_id


class TestKalmanTracking:
    def test_track_follows_moving_cluster(self, tracker):
        positions = [[0.0, 2.0, 1.0], [0.3, 2.0, 1.0], [0.6, 2.0, 1.0], [0.9, 2.0, 1.0]]
        active = []
        for pos in positions:
            active = tracker.update(make_person_points(pos, spread=0.02))
        assert len(active) == 1
        # 칼만 필터가 x 방향 이동을 따라가 마지막 위치 근처에 있어야 함
        assert active[0].position[0] == pytest.approx(0.9, abs=0.3)


class TestTrackTimeout:
    def test_track_removed_after_timeout(self, tracker):
        pts = make_person_points([0.0, 2.0, 1.0])
        for _ in range(3):
            tracker.update(pts)
        assert len(tracker._tracks) == 1

        time.sleep(0.3)  # track_timeout(0.2s) 초과
        active = tracker.update(np.zeros((0, 6), dtype=np.float32))
        assert active == []
        assert len(tracker._tracks) == 0


class TestFallStatusConsecutiveFrames:
    def test_fall_not_confirmed_below_threshold(self, tracker):
        pts = make_person_points([0.0, 2.0, 0.1])
        for _ in range(4):
            active = tracker.update(pts)
        pid = active[0].person_id

        tracker.update_fall_status(pid, True, 0.9)
        tracker.update_fall_status(pid, True, 0.9)
        # consecutive_frames=3 이므로 2번만으로는 아직 미확정
        assert tracker._tracks[pid].fall_detected is False

    def test_fall_confirmed_at_threshold(self, tracker):
        pts = make_person_points([0.0, 2.0, 0.1])
        for _ in range(4):
            active = tracker.update(pts)
        pid = active[0].person_id

        for _ in range(3):
            tracker.update_fall_status(pid, True, 0.9)
        assert tracker._tracks[pid].fall_detected is True
        assert pid in [t.person_id for t in tracker.fall_detected_persons]

    def test_fall_streak_resets_on_negative_frame(self, tracker):
        pts = make_person_points([0.0, 2.0, 0.1])
        for _ in range(4):
            active = tracker.update(pts)
        pid = active[0].person_id

        tracker.update_fall_status(pid, True, 0.9)
        tracker.update_fall_status(pid, True, 0.9)
        tracker.update_fall_status(pid, False, 0.0)  # streak 리셋
        tracker.update_fall_status(pid, True, 0.9)
        assert tracker._tracks[pid].fall_detected is False  # 다시 1번째

    def test_fall_cleared_when_no_longer_detected(self, tracker):
        pts = make_person_points([0.0, 2.0, 0.1])
        for _ in range(4):
            active = tracker.update(pts)
        pid = active[0].person_id

        for _ in range(3):
            tracker.update_fall_status(pid, True, 0.9)
        assert tracker._tracks[pid].fall_detected is True

        tracker.update_fall_status(pid, False, 0.0)
        assert tracker._tracks[pid].fall_detected is False
        assert tracker._tracks[pid].raw_fall_streak == 0

    def test_update_fall_status_on_unknown_person_is_noop(self, tracker):
        tracker.update_fall_status(999, True, 0.9)  # 존재하지 않는 ID - 예외 없이 무시


class TestPersonWindowHistory:
    def test_history_capped_at_16_frames(self, tracker):
        pts = make_person_points([0.0, 2.0, 1.0])
        active = []
        for _ in range(20):
            active = tracker.update(pts)
        pid = active[0].person_id
        history = tracker.get_person_window(pid)
        assert len(history) == 16

    def test_history_none_for_unknown_person(self, tracker):
        assert tracker.get_person_window(999) is None


class TestReset:
    def test_reset_clears_all_tracks(self, tracker):
        pts = make_person_points([0.0, 2.0, 1.0])
        for _ in range(3):
            tracker.update(pts)
        assert len(tracker._tracks) == 1

        tracker.reset()
        assert len(tracker._tracks) == 0
        assert tracker.active_persons == []
