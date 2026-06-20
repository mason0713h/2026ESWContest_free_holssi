"""
scripts/visualize_pointcloud.py
================================
포인트클라우드 시각화 디버깅 도구.

레이더에서 수신한 포인트클라우드를 실시간으로 시각화한다.
matplotlib 3D scatter plot으로 포인트 위치, 속도(색상), SNR(크기)을 표시.
Mock 모드에서 하드웨어 없이도 동작.

사용법:
    python scripts/visualize_pointcloud.py --mock
    python scripts/visualize_pointcloud.py --mock --scenario fall
    python scripts/visualize_pointcloud.py --save_gif output.gif
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_figure():
    """matplotlib 3D Figure 생성."""
    import matplotlib
    matplotlib.use("TkAgg" if sys.platform == "darwin" else "Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    fig = plt.figure(figsize=(12, 8))
    ax_3d = fig.add_subplot(121, projection="3d")
    ax_vel = fig.add_subplot(122)
    return fig, ax_3d, ax_vel


def update_3d_plot(ax, points: np.ndarray, frame_num: int, title: str = "") -> None:
    """3D scatter plot 업데이트."""
    ax.cla()
    if len(points) == 0:
        ax.set_title(f"Frame #{frame_num} - 포인트 없음")
        return

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    vel = points[:, 3] if points.shape[1] > 3 else np.zeros(len(points))
    snr = points[:, 4] if points.shape[1] > 4 else np.ones(len(points)) * 20

    # 속도를 색상으로, SNR을 크기로
    colors = vel
    sizes = np.clip((snr - snr.min()) / (snr.max() - snr.min() + 1e-6) * 50 + 10, 5, 60)

    sc = ax.scatter(x, y, z, c=colors, cmap="coolwarm", s=sizes, alpha=0.7)

    # 룸 경계 표시
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 5)
    ax.set_zlim(-0.5, 2.5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Frame #{frame_num} | N={len(points)} pts\n{title}")


def update_velocity_hist(ax, points: np.ndarray) -> None:
    """속도 히스토그램 업데이트."""
    ax.cla()
    if len(points) > 0 and points.shape[1] > 3:
        vel = points[:, 3]
        ax.hist(vel, bins=20, color="steelblue", edgecolor="white", alpha=0.7)
        ax.axvline(x=0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("도플러 속도 (m/s)")
        ax.set_ylabel("포인트 수")
        ax.set_title("속도 분포")
        ax.set_xlim(-3, 3)
    else:
        ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("속도 분포")


class PointCloudVisualizer:
    """
    실시간 포인트클라우드 시각화기.

    Args:
        mock_mode: True면 Mock 데이터 사용
        scenario: 'normal' | 'fall' | 'random' (Mock 모드)
        save_gif: GIF 저장 경로 (None이면 저장 안함)
        max_frames: 시각화할 최대 프레임 수
    """

    def __init__(
        self,
        mock_mode: bool = True,
        scenario: str = "random",
        save_gif: Optional[str] = None,
        max_frames: int = 100,
    ) -> None:
        self.mock_mode = mock_mode
        self.scenario = scenario
        self.save_gif = save_gif
        self.max_frames = max_frames

        self._frames: List[np.ndarray] = []
        self._frame_count = 0

    async def run(self) -> None:
        """시각화 루프 실행."""
        from radar.capture import RadarCapture
        from radar.preprocessor import PointCloudPreprocessor

        capture = RadarCapture(mock_mode=self.mock_mode, window_size=1, stride=1)
        preprocessor = PointCloudPreprocessor()

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax_3d, ax_vel = create_figure()
            frames_for_gif = []
            last_points = np.zeros((0, 6))

            scenario_weights = (
                [0.0, 1.0, 0.0] if self.scenario == "fall"
                else [1.0, 0.0, 0.0] if self.scenario == "normal"
                else [0.7, 0.25, 0.05]
            )

            async def on_window(window):
                nonlocal last_points
                self._frame_count += 1

                pc = window[0]
                if len(pc.points) > 0:
                    last_points = preprocessor.denoise(pc.points)
                else:
                    last_points = np.zeros((0, 6))

                # 실시간 출력
                status = "낙상" if (len(last_points) > 0 and last_points[:, 2].mean() < 0.3) else "정상"
                print(
                    f"\rFrame #{self._frame_count:4d} | "
                    f"포인트: {len(last_points):3d} | "
                    f"상태: {status} | "
                    f"z_mean: {last_points[:, 2].mean():.2f}m" if len(last_points) > 0 else "",
                    end="", flush=True,
                )

                # 그래프 업데이트 (매 5 프레임)
                if self._frame_count % 5 == 0:
                    update_3d_plot(ax_3d, last_points, self._frame_count, status)
                    update_velocity_hist(ax_vel, last_points)
                    fig.tight_layout()

                    if self.save_gif:
                        import io
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=60, bbox_inches="tight")
                        buf.seek(0)
                        frames_for_gif.append(buf.getvalue())

                if self._frame_count >= self.max_frames:
                    capture.stop()

            capture_task = asyncio.create_task(
                capture.capture_loop(
                    on_window,
                    mock_scenario_weights=scenario_weights,
                )
            )
            await asyncio.wait_for(capture_task, timeout=self.max_frames * 0.2)

        except asyncio.TimeoutError:
            pass
        except ImportError:
            print("matplotlib 미설치. 텍스트 모드로 출력합니다.\n")
            await self._text_mode_run(scenario_weights)
            return

        print(f"\n\n시각화 완료: {self._frame_count}프레임 처리")

        if self.save_gif and frames_for_gif:
            self._save_as_gif(frames_for_gif)

    async def _text_mode_run(self, scenario_weights: List[float]) -> None:
        """matplotlib 없이 텍스트로 포인트클라우드 출력."""
        from radar.capture import RadarCapture

        capture = RadarCapture(mock_mode=True, window_size=1, stride=1)
        count = 0

        async def on_window(window):
            nonlocal count
            count += 1
            pc = window[0]
            pts = pc.points
            if len(pts) > 0:
                z_mean = pts[:, 2].mean()
                vel_mean = pts[:, 3].mean()
                print(
                    f"Frame #{count:4d} | N={len(pts):3d} pts | "
                    f"z_mean={z_mean:.3f}m | vel_mean={vel_mean:+.3f}m/s"
                )
            else:
                print(f"Frame #{count:4d} | 포인트 없음")

            if count >= self.max_frames:
                capture.stop()

        await capture.capture_loop(on_window, mock_scenario_weights=scenario_weights)

    def _save_as_gif(self, frames: list) -> None:
        """프레임을 GIF로 저장."""
        try:
            from PIL import Image  # type: ignore
            import io
            pil_frames = [Image.open(io.BytesIO(f)) for f in frames]
            pil_frames[0].save(
                self.save_gif,
                save_all=True,
                append_images=pil_frames[1:],
                duration=200,
                loop=0,
            )
            print(f"GIF 저장 완료: {self.save_gif}")
        except ImportError:
            print("Pillow 미설치. GIF 저장 스킵")
        except Exception as e:
            print(f"GIF 저장 오류: {e}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description="포인트클라우드 시각화")
    parser.add_argument("--mock", action="store_true", default=True, help="Mock 모드 (기본 활성화)")
    parser.add_argument("--scenario", type=str, default="random",
                        choices=["normal", "fall", "random"],
                        help="Mock 시나리오")
    parser.add_argument("--max_frames", type=int, default=50, help="시각화 프레임 수")
    parser.add_argument("--save_gif", type=str, default=None, help="GIF 저장 경로")
    args = parser.parse_args()

    print(f"=== 포인트클라우드 시각화 ===")
    print(f"시나리오: {args.scenario}, 최대 프레임: {args.max_frames}\n")

    viz = PointCloudVisualizer(
        mock_mode=args.mock,
        scenario=args.scenario,
        save_gif=args.save_gif,
        max_frames=args.max_frames,
    )

    asyncio.run(viz.run())
