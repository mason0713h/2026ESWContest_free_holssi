"""
model/long_lie_detector.py
==========================
장시간 쓰러짐(Long-lie) 판별 모듈.

낙상 감지 후 일정 시간 이상 바닥에 머무는지 추적하여
단계적 알림 레벨을 상승시킨다.

알림 레벨:
    Level 0: 낙상 미감지 (정상)
    Level 1: 낙상 감지 (0~30초)
    Level 2: 30초 이상 쓰러짐 유지
    Level 3: 60초 이상 쓰러짐 유지
    Level 4: 120초 이상 쓰러짐 유지 (긴급)

바닥 판별 기준:
    추적 대상의 z 좌표 중앙값이 z_floor_threshold (0.3m) 이하
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 알림 레벨 정의
# ──────────────────────────────────────────────

class AlertLevel(IntEnum):
    """낙상/장시간 쓰러짐 알림 레벨."""
    NORMAL = 0       # 정상
    FALL = 1         # 낙상 감지 (즉시)
    LONG_LIE_30 = 2  # 30초 장기 쓰러짐
    LONG_LIE_60 = 3  # 60초 장기 쓰러짐
    LONG_LIE_120 = 4 # 120초 긴급


# ──────────────────────────────────────────────
# 개인별 Long-lie 추적 상태
# ──────────────────────────────────────────────

@dataclass
class LongLieState:
    """
    개인별 장시간 쓰러짐 추적 상태.

    Attributes:
        person_id: 대상 인원 ID
        alert_level: 현재 알림 레벨
        fall_start_time: 낙상 최초 감지 시각
        last_z_median: 최근 z 좌표 중앙값 (바닥 여부 판별)
        level_triggered: 각 레벨 알림 발생 여부
    """
    person_id: int
    alert_level: AlertLevel = AlertLevel.NORMAL
    fall_start_time: Optional[float] = None
    last_z_median: float = 1.0
    level_triggered: Dict[int, bool] = field(
        default_factory=lambda: {1: False, 2: False, 3: False, 4: False}
    )

    @property
    def fall_duration(self) -> float:
        """낙상 감지 후 경과 시간 (초)."""
        if self.fall_start_time is None:
            return 0.0
        return time.time() - self.fall_start_time

    def reset(self) -> None:
        """상태 초기화 (낙상 해제)."""
        self.alert_level = AlertLevel.NORMAL
        self.fall_start_time = None
        self.level_triggered = {1: False, 2: False, 3: False, 4: False}


class LongLieDetector:
    """
    장시간 쓰러짐 감지기.

    낙상이 감지된 후 시간 경과에 따라 알림 레벨을 상승시킨다.
    각 단계별 콜백 함수를 등록하여 알림 발송 로직을 연결할 수 있다.

    Args:
        z_floor_threshold: 바닥 높이 기준 (m). 이 값 이하면 바닥으로 판단.
        level1_seconds: 1→2단계 상승 시간 (초) (기본: 30)
        level2_seconds: 2→3단계 상승 시간 (초) (기본: 60)
        level3_seconds: 3→4단계 상승 시간 (초) (기본: 120)
        recovery_z_threshold: 자력 회복으로 판단하는 z 좌표 최솟값 (m)
        recovery_hold_seconds: 회복 판정 유지 시간 (초)
    """

    def __init__(
        self,
        z_floor_threshold: float = 0.3,
        level1_seconds: float = 30.0,
        level2_seconds: float = 60.0,
        level3_seconds: float = 120.0,
        recovery_z_threshold: float = 0.5,
        recovery_hold_seconds: float = 5.0,
    ) -> None:
        self.z_floor_threshold = z_floor_threshold
        self.level1_seconds = level1_seconds
        self.level2_seconds = level2_seconds
        self.level3_seconds = level3_seconds
        self.recovery_z_threshold = recovery_z_threshold
        self.recovery_hold_seconds = recovery_hold_seconds

        # 단계별 상승 임계값 (초)
        self._level_thresholds = {
            AlertLevel.LONG_LIE_30: level1_seconds,
            AlertLevel.LONG_LIE_60: level2_seconds,
            AlertLevel.LONG_LIE_120: level3_seconds,
        }

        # 개인별 상태
        self._states: Dict[int, LongLieState] = {}

        # 회복 추적
        self._recovery_start: Dict[int, float] = {}

        # 콜백 등록
        self._callbacks: Dict[AlertLevel, List[Callable]] = {
            lvl: [] for lvl in AlertLevel
        }

        logger.info(
            "LongLieDetector 초기화: z_threshold=%.2f, levels=[%ds, %ds, %ds]",
            z_floor_threshold,
            level1_seconds,
            level2_seconds,
            level3_seconds,
        )

    def register_callback(
        self, level: AlertLevel, callback: Callable
    ) -> None:
        """
        특정 레벨 도달 시 실행할 콜백 등록.

        Args:
            level: 콜백을 발동시킬 AlertLevel
            callback: callback(person_id, state) 형식의 callable
        """
        self._callbacks[level].append(callback)
        logger.debug("콜백 등록: level=%s, callback=%s", level.name, callback.__name__)

    def _get_state(self, person_id: int) -> LongLieState:
        """개인별 상태 가져오기 (없으면 생성)."""
        if person_id not in self._states:
            self._states[person_id] = LongLieState(person_id=person_id)
        return self._states[person_id]

    def _is_on_floor(self, points: np.ndarray) -> bool:
        """
        포인트 배열에서 바닥에 누워있는지 판별.

        z 좌표 중앙값이 threshold 이하이고
        z 표준편차가 작으면 (넓게 퍼진 상태 = 누워있음) 바닥으로 판단.

        Args:
            points: (N, 6) 포인트 배열

        Returns:
            바닥 여부
        """
        if len(points) == 0:
            return False

        z_values = points[:, 2]
        z_median = float(np.median(z_values))
        z_std = float(np.std(z_values))

        # 바닥 조건: z 중앙값이 낮고 (누워있음), z 표준편차가 크지 않음
        is_floor = z_median < self.z_floor_threshold

        logger.debug(
            "바닥 판별: z_median=%.3f, z_std=%.3f, is_floor=%s",
            z_median, z_std, is_floor
        )
        return is_floor

    def update(
        self,
        person_id: int,
        fall_detected: bool,
        fall_confidence: float,
        person_points: Optional[np.ndarray] = None,
    ) -> AlertLevel:
        """
        낙상 감지 상태 업데이트 및 알림 레벨 반환.

        Args:
            person_id: 대상 인원 ID
            fall_detected: 낙상 감지 여부
            fall_confidence: 낙상 감지 신뢰도
            person_points: 해당 인원의 포인트 배열 (바닥 판별용)

        Returns:
            현재 AlertLevel
        """
        state = self._get_state(person_id)
        now = time.time()

        if not fall_detected:
            # 낙상 미감지 → 회복 판별
            if state.alert_level > AlertLevel.NORMAL:
                # 회복 여부 체크 (z 좌표 상승 확인)
                if person_points is not None and not self._is_on_floor(person_points):
                    if person_id not in self._recovery_start:
                        self._recovery_start[person_id] = now
                    elif now - self._recovery_start[person_id] >= self.recovery_hold_seconds:
                        # 자력 회복 확인
                        logger.info("인원 #%d 자력 회복 감지", person_id)
                        prev_level = state.alert_level
                        state.reset()
                        if person_id in self._recovery_start:
                            del self._recovery_start[person_id]
                        self._fire_callbacks(AlertLevel.NORMAL, person_id, state)
                        return AlertLevel.NORMAL
                else:
                    # 아직 바닥에 있음 → 회복 타이머 리셋
                    if person_id in self._recovery_start:
                        del self._recovery_start[person_id]
            return state.alert_level

        # 낙상 감지
        if person_id in self._recovery_start:
            del self._recovery_start[person_id]

        # 최초 낙상 감지
        if state.fall_start_time is None:
            state.fall_start_time = now
            state.alert_level = AlertLevel.FALL
            if not state.level_triggered[1]:
                state.level_triggered[1] = True
                logger.warning(
                    "인원 #%d 낙상 감지! 신뢰도=%.3f", person_id, fall_confidence
                )
                self._fire_callbacks(AlertLevel.FALL, person_id, state)

        # 장기 쓰러짐 레벨 상승 체크
        duration = state.fall_duration
        new_level = state.alert_level

        if duration >= self.level3_seconds and not state.level_triggered[4]:
            new_level = AlertLevel.LONG_LIE_120
            state.level_triggered[4] = True
            logger.warning(
                "인원 #%d 장시간 쓰러짐 (%.0f초 경과) - Level 4 긴급!",
                person_id, duration
            )
            self._fire_callbacks(AlertLevel.LONG_LIE_120, person_id, state)

        elif duration >= self.level2_seconds and not state.level_triggered[3]:
            new_level = AlertLevel.LONG_LIE_60
            state.level_triggered[3] = True
            logger.warning(
                "인원 #%d 장시간 쓰러짐 (%.0f초 경과) - Level 3",
                person_id, duration
            )
            self._fire_callbacks(AlertLevel.LONG_LIE_60, person_id, state)

        elif duration >= self.level1_seconds and not state.level_triggered[2]:
            new_level = AlertLevel.LONG_LIE_30
            state.level_triggered[2] = True
            logger.warning(
                "인원 #%d 장시간 쓰러짐 (%.0f초 경과) - Level 2",
                person_id, duration
            )
            self._fire_callbacks(AlertLevel.LONG_LIE_30, person_id, state)

        state.alert_level = new_level
        return state.alert_level

    def _fire_callbacks(
        self,
        level: AlertLevel,
        person_id: int,
        state: LongLieState,
    ) -> None:
        """등록된 콜백 실행."""
        for cb in self._callbacks.get(level, []):
            try:
                cb(person_id, state)
            except Exception as e:
                logger.error("콜백 실행 오류 (level=%s): %s", level.name, e)

    def get_state(self, person_id: int) -> Optional[LongLieState]:
        """특정 인원의 상태 조회."""
        return self._states.get(person_id)

    def clear_person(self, person_id: int) -> None:
        """특정 인원의 상태 삭제."""
        self._states.pop(person_id, None)
        self._recovery_start.pop(person_id, None)

    @property
    def all_alerts(self) -> Dict[int, AlertLevel]:
        """현재 알림 레벨이 FALL 이상인 모든 인원."""
        return {
            pid: state.alert_level
            for pid, state in self._states.items()
            if state.alert_level > AlertLevel.NORMAL
        }

    def summary(self) -> str:
        """현재 상태 요약 문자열."""
        if not self._states:
            return "추적 중인 인원 없음"
        lines = []
        for pid, state in self._states.items():
            lines.append(
                f"  인원 #{pid}: {state.alert_level.name} "
                f"(경과: {state.fall_duration:.1f}초)"
            )
        return "\n".join(lines)


# ──────────────────────────────────────────────
# 단독 실행 테스트
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import time

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=== LongLieDetector 테스트 ===\n")

    alert_log = []

    def on_fall(person_id, state):
        alert_log.append(f"[{time.strftime('%H:%M:%S')}] 인원 #{person_id} 낙상 감지!")
        print(f"  🔔 콜백: 인원 #{person_id} 낙상 감지!")

    def on_long_lie_30(person_id, state):
        alert_log.append(f"[{time.strftime('%H:%M:%S')}] 인원 #{person_id} 30초 장기 쓰러짐!")
        print(f"  🔔 콜백: 인원 #{person_id} 30초 장기 쓰러짐!")

    def on_long_lie_60(person_id, state):
        alert_log.append(f"[{time.strftime('%H:%M:%S')}] 인원 #{person_id} 60초 장기 쓰러짐!")
        print(f"  🔔 콜백: 인원 #{person_id} 60초 장기 쓰러짐!")

    # 테스트용 짧은 시간 임계값
    detector = LongLieDetector(
        z_floor_threshold=0.3,
        level1_seconds=3.0,    # 테스트용: 3초
        level2_seconds=6.0,    # 테스트용: 6초
        level3_seconds=10.0,   # 테스트용: 10초
    )

    detector.register_callback(AlertLevel.FALL, on_fall)
    detector.register_callback(AlertLevel.LONG_LIE_30, on_long_lie_30)
    detector.register_callback(AlertLevel.LONG_LIE_60, on_long_lie_60)

    rng = np.random.default_rng(42)

    # 낙상 시뮬레이션 (인원 #1)
    print("인원 #1 낙상 시뮬레이션 시작...")
    fall_points = np.zeros((30, 6))
    fall_points[:, 2] = rng.uniform(0.05, 0.2, 30)  # z ≈ 0 (바닥)

    for i in range(15):
        level = detector.update(
            person_id=1,
            fall_detected=True,
            fall_confidence=0.92,
            person_points=fall_points,
        )
        print(f"  t+{i}s: Level={level.name}, 경과={detector.get_state(1).fall_duration:.1f}초")
        time.sleep(1)

    print(f"\n전체 알림 로그:")
    for log in alert_log:
        print(f"  {log}")

    print(f"\n현재 활성 알림: {detector.all_alerts}")
    print("\nLongLieDetector 테스트 완료!")
