"""
scripts/collect_data.py
=======================
학습 데이터 수집 스크립트.

TI IWR6843 레이더에서 포인트클라우드 데이터를 수집하고
라벨(낙상/정상)과 함께 .npy 파일로 저장한다.

수집 절차:
  1. 피험자를 레이더 앞에 세움
  2. 키보드로 레이블 지정 (f=fall, n=normal, s=stop)
  3. 설정한 시간 동안 포인트클라우드 시퀀스 수집
  4. 슬라이딩 윈도우로 분할하여 저장

사용법:
    python scripts/collect_data.py --output data/ --mock
    python scripts/collect_data.py --output data/ --label fall --duration 30
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from radar.capture import RadarCapture, PointCloud
from radar.preprocessor import PointCloudPreprocessor

logger = logging.getLogger(__name__)


class DataCollector:
    """
    레이더 데이터 수집기.

    Args:
        output_dir: 저장 디렉토리
        window_size: 윈도우 크기
        target_points: 정규화 포인트 수
        mock_mode: Mock 모드
    """

    def __init__(
        self,
        output_dir: str,
        window_size: int = 16,
        target_points: int = 64,
        mock_mode: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.window_size = window_size
        self.mock_mode = mock_mode

        self.capture = RadarCapture(
            window_size=window_size, stride=window_size, mock_mode=mock_mode
        )
        self.preprocessor = PointCloudPreprocessor(target_points=target_points)

        self._current_label: str = "normal"
        self._windows: list = []
        self._collecting = False
        self._saved_count = {"fall": 0, "normal": 0}

    async def collect_session(
        self,
        label: str,
        duration: float,
        session_name: str = "",
    ) -> int:
        """
        단일 세션 데이터 수집.

        Args:
            label: 라벨 ('fall' 또는 'normal')
            duration: 수집 시간 (초)
            session_name: 세션 이름 (파일명 접두사)

        Returns:
            저장된 윈도우 수
        """
        self._current_label = label
        self._windows.clear()
        self._collecting = True

        label_dir = self.output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{label.upper()}] 데이터 수집 시작 ({duration:.0f}초)...")
        print("레이더 앞에서 시작하세요.")

        # 3초 카운트다운
        for i in range(3, 0, -1):
            print(f"  {i}...")
            await asyncio.sleep(1)
        print("  수집 시작!")

        collected_windows = []

        async def on_window(window):
            tensor = self.preprocessor.process_window(window)
            collected_windows.append(tensor)

        # 수집 태스크
        capture_task = asyncio.create_task(
            self.capture.capture_loop(on_window)
        )

        # 지정 시간 동안 수집
        await asyncio.sleep(duration)
        self.capture.stop()
        capture_task.cancel()
        try:
            await asyncio.wait_for(capture_task, timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

        # 저장
        saved = 0
        ts = int(time.time())
        prefix = session_name or f"{label}_{ts}"
        for i, tensor in enumerate(collected_windows):
            fname = label_dir / f"{prefix}_{i+1:04d}.npy"
            np.save(str(fname), tensor.astype(np.float32))
            saved += 1
            self._saved_count[label] += 1

        print(f"  {saved}개 윈도우 저장 완료 → {label_dir}")
        return saved

    def print_stats(self) -> None:
        """수집 통계 출력."""
        print("\n=== 수집 통계 ===")
        for label, count in self._saved_count.items():
            print(f"  {label}: {count}개")
        print(f"  저장 위치: {self.output_dir}")


async def interactive_collect(args: argparse.Namespace) -> None:
    """대화형 수집 모드."""
    collector = DataCollector(
        output_dir=args.output,
        mock_mode=args.mock,
    )

    if not args.mock:
        print("레이더 연결 중...")
        collector.capture.connect()

    print("\n=== Fall Guardian 데이터 수집 도구 ===")
    print("라벨: [f]all / [n]ormal / [q]uit")

    while True:
        cmd = input("\n명령 입력 (f/n/q): ").strip().lower()
        if cmd == "q":
            break
        elif cmd in ("f", "n"):
            label = "fall" if cmd == "f" else "normal"
            try:
                duration = float(input(f"  수집 시간 (초) [기본 15]: ").strip() or "15")
                session = input("  세션 이름 (엔터=자동): ").strip()
            except (ValueError, EOFError):
                duration = 15.0
                session = ""
            await collector.collect_session(label, duration, session)
        else:
            print("  알 수 없는 명령 (f/n/q만 사용 가능)")

    collector.print_stats()
    if not args.mock:
        collector.capture.disconnect()


async def batch_collect(args: argparse.Namespace) -> None:
    """배치 수집 모드 (--label 지정)."""
    collector = DataCollector(
        output_dir=args.output,
        mock_mode=args.mock,
    )

    if not args.mock:
        collector.capture.connect()

    await collector.collect_session(
        label=args.label,
        duration=args.duration,
        session_name=args.session,
    )
    collector.print_stats()

    if not args.mock:
        collector.capture.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Fall Guardian 데이터 수집")
    parser.add_argument("--output", type=str, default="data/", help="저장 디렉토리")
    parser.add_argument("--mock", action="store_true", help="Mock 모드")
    parser.add_argument("--label", type=str, choices=["fall", "normal"], help="배치 수집 라벨")
    parser.add_argument("--duration", type=float, default=15.0, help="수집 시간 (초)")
    parser.add_argument("--session", type=str, default="", help="세션 이름")
    args = parser.parse_args()

    print("=== 데이터 수집 스크립트 ===")
    if args.label:
        asyncio.run(batch_collect(args))
    else:
        asyncio.run(interactive_collect(args))
