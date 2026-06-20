"""
radar/capture.py
================
TI IWR6843ISK-ODS 60GHz 4D mmWave 레이더 데이터 캡처 모듈.

시리얼 UART를 통해 레이더로부터 TLV(Type-Length-Value) 패킷을 수신하고,
Enhanced Point Cloud (TLV 타입 1006) 데이터를 파싱하여 포인트클라우드를
슬라이딩 윈도우 버퍼로 제공한다.

하드웨어 없이 실행 시 mock_mode=True로 랜덤 포인트클라우드를 생성한다.

참고:
    - TI mmWave SDK TLV Format: MMWAVE SDK User Guide
    - IWR6843ISK-ODS EVM 데이터시트
"""

from __future__ import annotations

import asyncio
import logging
import random
import struct
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 상수 정의
# ──────────────────────────────────────────────
MAGIC_WORD = bytes([2, 1, 4, 3, 6, 5, 8, 7])
FRAME_HEADER_SIZE = 40          # TI SDK 프레임 헤더 크기 (bytes)
TLV_HEADER_SIZE = 8             # TLV 헤더 크기 (bytes)
TLV_TYPE_ENHANCED_POINT_CLOUD = 1006
POINT_RECORD_SIZE = 20          # Enhanced PointCloud 포인트 레코드 크기 (bytes)
                                 # x(4) + y(4) + z(4) + velocity(4) + snr(2) + noise(2)


# ──────────────────────────────────────────────
# 데이터 클래스
# ──────────────────────────────────────────────

@dataclass
class PointCloud:
    """
    단일 레이더 프레임의 포인트클라우드 데이터.

    Attributes:
        timestamp: 수신 시각 (Unix 타임스탬프)
        frame_number: 레이더 내부 프레임 번호
        points: (N, 6) 형태의 배열 [x, y, z, velocity, snr, noise]
        num_points: 유효 포인트 수
    """
    timestamp: float
    frame_number: int
    points: np.ndarray          # shape: (N, 6)
    num_points: int = field(init=False)

    def __post_init__(self) -> None:
        self.num_points = len(self.points)

    @property
    def xyz(self) -> np.ndarray:
        """xyz 좌표만 추출 (N, 3)."""
        return self.points[:, :3]

    @property
    def velocity(self) -> np.ndarray:
        """도플러 속도 추출 (N,)."""
        return self.points[:, 3]

    @property
    def snr(self) -> np.ndarray:
        """SNR 추출 (N,)."""
        return self.points[:, 4]


@dataclass
class FrameWindow:
    """
    슬라이딩 윈도우 버퍼 (window_size 개의 프레임).

    Attributes:
        frames: 포인트클라우드 리스트
        window_size: 윈도우 크기
        stride: 슬라이딩 스트라이드
    """
    window_size: int
    stride: int
    frames: Deque[PointCloud] = field(default_factory=deque)
    _frame_count: int = field(default=0, init=False, repr=False)

    def push(self, pc: PointCloud) -> Optional[List[PointCloud]]:
        """
        프레임을 버퍼에 추가하고, 윈도우가 가득 차면 슬라이딩 윈도우 반환.

        Args:
            pc: 추가할 포인트클라우드 프레임

        Returns:
            윈도우가 가득 찬 경우 window_size 크기의 프레임 리스트,
            아직 부족하면 None
        """
        self.frames.append(pc)
        self._frame_count += 1

        if len(self.frames) > self.window_size:
            self.frames.popleft()

        if len(self.frames) == self.window_size:
            if (self._frame_count - self.window_size) % self.stride == 0:
                return list(self.frames)
        return None


# ──────────────────────────────────────────────
# 메인 캡처 클래스
# ──────────────────────────────────────────────

class RadarCapture:
    """
    TI IWR6843ISK-ODS 레이더 데이터 캡처 및 파싱 클래스.

    시리얼 포트 2개(설정 포트 + 데이터 포트)를 사용하여 레이더를 초기화하고,
    실시간으로 TLV 패킷을 수신·파싱한다.

    Args:
        config_port: CLI 설정 시리얼 포트 경로 (예: /dev/ttyUSB0)
        data_port: 데이터 시리얼 포트 경로 (예: /dev/ttyUSB1)
        config_baudrate: 설정 포트 보드레이트
        data_baudrate: 데이터 포트 보드레이트
        window_size: 슬라이딩 윈도우 프레임 수
        stride: 슬라이딩 윈도우 스트라이드
        mock_mode: True면 실제 하드웨어 없이 랜덤 데이터 생성
    """

    def __init__(
        self,
        config_port: str = "/dev/ttyUSB0",
        data_port: str = "/dev/ttyUSB1",
        config_baudrate: int = 115200,
        data_baudrate: int = 921600,
        window_size: int = 16,
        stride: int = 8,
        mock_mode: bool = False,
        cfg_file: Optional[str] = None,
    ) -> None:
        self.config_port = config_port
        self.data_port = data_port
        self.config_baudrate = config_baudrate
        self.data_baudrate = data_baudrate
        self.window_size = window_size
        self.stride = stride
        self.mock_mode = mock_mode
        # mmWave Demo Visualizer로 생성한 실제 .cfg 파일 경로.
        # 지정하지 않으면 내장 기본 명령어를 사용한다 (참고용, 실제 장비
        # 캘리브레이션값과 다를 수 있음 — 가능하면 항상 .cfg 파일을 사용할 것).
        self.cfg_file = cfg_file

        self._ser_config = None   # 설정 포트 (pyserial)
        self._ser_data = None     # 데이터 포트 (pyserial)
        self._buffer = bytearray()
        self._frame_window = FrameWindow(window_size=window_size, stride=stride)
        self._frame_number = 0
        self._running = False

        logger.info(
            "RadarCapture 초기화: mock_mode=%s, window_size=%d, stride=%d",
            mock_mode, window_size, stride,
        )

    # ── 초기화 및 종료 ─────────────────────────

    def connect(self) -> None:
        """
        시리얼 포트 연결 및 레이더 설정 전송.
        mock_mode=True 이면 스킵.
        """
        if self.mock_mode:
            logger.info("[Mock] 레이더 연결 스킵")
            return

        try:
            import serial  # type: ignore
        except ImportError:
            raise ImportError("pyserial 패키지가 필요합니다: pip install pyserial")

        logger.info("레이더 시리얼 포트 연결 중: %s, %s", self.config_port, self.data_port)
        self._ser_config = serial.Serial(
            self.config_port, self.config_baudrate, timeout=1
        )
        self._ser_data = serial.Serial(
            self.data_port, self.data_baudrate, timeout=0.1
        )
        self._send_radar_config()
        logger.info("레이더 연결 완료")

    def disconnect(self) -> None:
        """시리얼 포트 닫기."""
        self._running = False
        if self._ser_config and self._ser_config.is_open:
            self._ser_config.close()
        if self._ser_data and self._ser_data.is_open:
            self._ser_data.close()
        logger.info("레이더 연결 해제")

    def _load_cfg_commands(self) -> List[str]:
        """
        설정 명령어 목록을 로드.

        cfg_file이 지정되어 있으면 TI mmWave Demo Visualizer로 생성한
        실제 .cfg 파일을 읽어 그대로 사용한다 (빈 줄/주석 줄 제외).
        지정되지 않으면 내장 기본값을 폴백으로 사용한다.
        """
        if self.cfg_file:
            cfg_path = Path(self.cfg_file)
            if not cfg_path.exists():
                raise FileNotFoundError(f".cfg 파일을 찾을 수 없음: {self.cfg_file}")
            lines = []
            with open(cfg_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("%"):
                        continue
                    lines.append(line + "\n")
            logger.info(".cfg 파일 로드: %s (%d개 명령)", self.cfg_file, len(lines))
            return lines

        logger.warning(
            "cfg_file 미지정 - 내장 기본 명령어 사용 (실제 장비 특성과 다를 수 있음, "
            "radar.cfg_file 설정 권장)"
        )
        return [
            "sensorStop\n",
            "flushCfg\n",
            # 기본 프로파일 설정 (60GHz, range 5m, velocity ±3m/s)
            "profileCfg 0 60 411 7 57 0 0 70 1 256 5209 0 0 158\n",
            "chirpCfg 0 0 0 0 0 0 0 1\n",
            "chirpCfg 1 1 0 0 0 0 0 2\n",
            "chirpCfg 2 2 0 0 0 0 0 4\n",
            "frameCfg 0 2 16 0 100 1 0\n",
            "guiMonitor -1 1 0 0 0 0 0\n",
            "cfarCfg -1 0 2 8 4 3 0 15.0 0\n",
            "cfarCfg -1 1 0 4 2 3 1 15.0 0\n",
            "multiObjBeamForming -1 1 0.5\n",
            "calibDcRangeSig -1 0 -5 8 256\n",
            "extendedMaxVelocity -1 0\n",
            "sensorStart\n",
        ]

    def _send_radar_config(self) -> None:
        """
        레이더에 설정 명령어를 한 줄씩 전송하고, 각 명령에 대해
        CLI 포트가 회신하는 "Done"/에러 응답을 확인한다.

        TI mmWave SDK의 CLI는 명령 1개당 응답 1개를 보장하므로,
        응답을 기다리지 않고 다음 명령을 보내면 버퍼 오버플로우로
        설정이 누락될 수 있다 (실제 하드웨어에서 빈번히 발생하는 문제).
        """
        commands = self._load_cfg_commands()
        for cmd in commands:
            self._ser_config.reset_input_buffer()
            self._ser_config.write(cmd.encode())
            response = self._wait_for_cli_response(timeout=1.0)
            if "Done" not in response and cmd.strip() not in ("sensorStop",):
                logger.warning(
                    "레이더 CLI 응답 비정상 (cmd=%r): %r", cmd.strip(), response.strip()
                )
        logger.info("레이더 설정 전송 완료 (%d 명령)", len(commands))

    def _wait_for_cli_response(self, timeout: float = 1.0) -> str:
        """CLI 포트에서 명령 응답("Done" 또는 에러 메시지)을 읽는다."""
        deadline = time.time() + timeout
        buf = b""
        while time.time() < deadline:
            chunk = self._ser_config.read(256)
            if chunk:
                buf += chunk
                if b"Done" in buf or b"Error" in buf:
                    break
            else:
                time.sleep(0.01)
        return buf.decode(errors="replace")

    # ── TLV 파싱 ───────────────────────────────

    def _find_magic_word(self, data: bytearray) -> int:
        """
        바이트 배열에서 MAGIC_WORD 위치 검색.

        Returns:
            MAGIC_WORD 시작 인덱스, 없으면 -1
        """
        for i in range(len(data) - len(MAGIC_WORD) + 1):
            if data[i: i + len(MAGIC_WORD)] == MAGIC_WORD:
                return i
        return -1

    def _parse_frame_header(self, data: bytes) -> Optional[dict]:
        """
        TI SDK 프레임 헤더(40 bytes) 파싱.

        Frame Header Layout (TI mmWave SDK):
            magic_word(8) + version(4) + total_packet_len(4) +
            platform(4) + frame_number(4) + cpu_cycles(4) +
            num_objects(4) + num_tlvs(4) + subframe_idx(4)

        Returns:
            헤더 딕셔너리 또는 None (크기 부족)
        """
        if len(data) < FRAME_HEADER_SIZE:
            return None

        try:
            magic = data[:8]
            if magic != MAGIC_WORD:
                return None

            (
                version,
                total_len,
                platform,
                frame_num,
                cpu_cycles,
                num_objects,
                num_tlvs,
                subframe,
            ) = struct.unpack_from("<IIIIIIII", data, 8)

            return {
                "version": version,
                "total_len": total_len,
                "platform": platform,
                "frame_number": frame_num,
                "cpu_cycles": cpu_cycles,
                "num_objects": num_objects,
                "num_tlvs": num_tlvs,
                "subframe_idx": subframe,
            }
        except struct.error as e:
            logger.debug("프레임 헤더 파싱 오류: %s", e)
            return None

    def _parse_tlv_header(self, data: bytes, offset: int) -> Optional[Tuple[int, int]]:
        """
        TLV 헤더(8 bytes) 파싱.

        Returns:
            (tlv_type, tlv_length) 또는 None
        """
        if offset + TLV_HEADER_SIZE > len(data):
            return None
        try:
            tlv_type, tlv_length = struct.unpack_from("<II", data, offset)
            return tlv_type, tlv_length
        except struct.error:
            return None

    def _parse_enhanced_point_cloud(
        self, data: bytes, offset: int, length: int
    ) -> np.ndarray:
        """
        Enhanced Point Cloud TLV(타입 1006) 파싱.

        포인트 레코드 형식 (20 bytes each):
            x(f32) + y(f32) + z(f32) + velocity(f32) + snr(u16) + noise(u16)

        Args:
            data: 전체 패킷 바이트
            offset: TLV 데이터 시작 위치
            length: TLV 데이터 길이 (bytes)

        Returns:
            (N, 6) 형태의 포인트 배열 [x, y, z, vel, snr, noise]
        """
        num_points = length // POINT_RECORD_SIZE
        if num_points == 0:
            return np.zeros((0, 6), dtype=np.float32)

        points = []
        for i in range(num_points):
            base = offset + i * POINT_RECORD_SIZE
            if base + POINT_RECORD_SIZE > len(data):
                break
            x, y, z, vel = struct.unpack_from("<ffff", data, base)
            snr, noise = struct.unpack_from("<HH", data, base + 16)
            points.append([x, y, z, vel, float(snr) * 0.1, float(noise) * 0.1])

        return np.array(points, dtype=np.float32) if points else np.zeros((0, 6), dtype=np.float32)

    def parse_packet(self, raw: bytes) -> Optional[PointCloud]:
        """
        수신된 raw 바이트 패킷을 PointCloud로 변환.

        Args:
            raw: MAGIC_WORD부터 시작하는 완전한 패킷 바이트

        Returns:
            파싱 성공 시 PointCloud, 실패 시 None
        """
        header = self._parse_frame_header(raw)
        if header is None:
            return None

        offset = FRAME_HEADER_SIZE
        all_points = []

        for _ in range(header["num_tlvs"]):
            tlv_hdr = self._parse_tlv_header(raw, offset)
            if tlv_hdr is None:
                break
            tlv_type, tlv_length = tlv_hdr
            offset += TLV_HEADER_SIZE

            if tlv_type == TLV_TYPE_ENHANCED_POINT_CLOUD:
                pts = self._parse_enhanced_point_cloud(raw, offset, tlv_length)
                all_points.append(pts)

            offset += tlv_length

        if all_points:
            combined = np.vstack(all_points)
        else:
            combined = np.zeros((0, 6), dtype=np.float32)

        return PointCloud(
            timestamp=time.time(),
            frame_number=header["frame_number"],
            points=combined,
        )

    # ── Mock 데이터 생성 ────────────────────────

    def _generate_mock_frame(self, scenario: str = "normal") -> PointCloud:
        """
        실제 하드웨어 없이 가상 포인트클라우드 생성.

        Args:
            scenario: 'normal' | 'fall' | 'empty'

        Returns:
            가상 PointCloud
        """
        rng = np.random.default_rng(seed=None)

        if scenario == "fall":
            # 낙상: 포인트가 바닥(z≈0)에 넓게 퍼지고 속도 급변
            n = rng.integers(20, 40)
            x = rng.normal(0.0, 0.5, n).astype(np.float32)
            y = rng.normal(2.0, 0.4, n).astype(np.float32)
            z = rng.normal(0.1, 0.15, n).astype(np.float32)   # 바닥 근처
            vel = rng.normal(-1.5, 0.5, n).astype(np.float32) # 하강 속도
        elif scenario == "empty":
            n = 0
            x = y = z = vel = np.array([], dtype=np.float32)
        else:  # normal (서있음)
            n = rng.integers(30, 60)
            x = rng.normal(0.0, 0.3, n).astype(np.float32)
            y = rng.normal(2.0, 0.3, n).astype(np.float32)
            z = rng.normal(1.0, 0.3, n).astype(np.float32)    # 서있는 높이
            vel = rng.normal(0.0, 0.1, n).astype(np.float32)  # 거의 정지

        if n > 0:
            snr = rng.uniform(10, 40, n).astype(np.float32)
            noise = rng.uniform(1, 5, n).astype(np.float32)
            points = np.column_stack([x, y, z, vel, snr, noise])
        else:
            points = np.zeros((0, 6), dtype=np.float32)

        self._frame_number += 1
        return PointCloud(
            timestamp=time.time(),
            frame_number=self._frame_number,
            points=points,
        )

    # ── 비동기 캡처 루프 ────────────────────────

    async def capture_loop(
        self,
        callback,
        mock_scenario_weights: Optional[List[float]] = None,
    ) -> None:
        """
        비동기 캡처 루프. 프레임을 수신하고 슬라이딩 윈도우가 완성되면
        callback(window: List[PointCloud])을 호출한다.

        Args:
            callback: 윈도우 완성 시 호출할 비동기 함수
            mock_scenario_weights: mock 모드에서 [normal, fall, empty] 가중치
        """
        self._running = True
        frame_period = 1.0 / 10  # 10 Hz

        if mock_scenario_weights is None:
            mock_scenario_weights = [0.85, 0.10, 0.05]

        logger.info("캡처 루프 시작 (mock=%s)", self.mock_mode)

        while self._running:
            loop_start = asyncio.get_event_loop().time()

            if self.mock_mode:
                scenario = random.choices(
                    ["normal", "fall", "empty"],
                    weights=mock_scenario_weights,
                )[0]
                pc = self._generate_mock_frame(scenario)
                await asyncio.sleep(frame_period)
            else:
                pc = await self._read_frame_serial()

            if pc is not None:
                window = self._frame_window.push(pc)
                if window is not None:
                    await callback(window)

            elapsed = asyncio.get_event_loop().time() - loop_start
            sleep_time = max(0.0, frame_period - elapsed)
            if sleep_time > 0 and not self.mock_mode:
                await asyncio.sleep(sleep_time)

    async def _read_frame_serial(self) -> Optional[PointCloud]:
        """
        시리얼 포트에서 완전한 프레임 패킷을 읽어 파싱.

        Returns:
            파싱된 PointCloud 또는 None
        """
        loop = asyncio.get_event_loop()

        try:
            # 논블로킹으로 시리얼 데이터 읽기 (executor 사용)
            chunk = await loop.run_in_executor(
                None, lambda: self._ser_data.read(4096)
            )
            if chunk:
                self._buffer.extend(chunk)

            # 매직 워드 탐색
            idx = self._find_magic_word(self._buffer)
            if idx < 0:
                if len(self._buffer) > 4096:
                    self._buffer = self._buffer[-len(MAGIC_WORD):]
                return None

            # 버퍼 정렬
            if idx > 0:
                self._buffer = self._buffer[idx:]

            # 헤더 파싱으로 전체 패킷 길이 확인
            if len(self._buffer) < FRAME_HEADER_SIZE:
                return None

            header = self._parse_frame_header(bytes(self._buffer))
            if header is None:
                self._buffer = self._buffer[1:]
                return None

            total_len = header["total_len"]
            if len(self._buffer) < total_len:
                return None  # 아직 완전한 패킷 없음

            packet = bytes(self._buffer[:total_len])
            self._buffer = self._buffer[total_len:]
            return self.parse_packet(packet)

        except Exception as e:
            logger.error("시리얼 읽기 오류: %s", e)
            return None

    def stop(self) -> None:
        """캡처 루프 중지."""
        self._running = False
        logger.info("캡처 루프 중지 요청")


# ──────────────────────────────────────────────
# 단독 실행 테스트
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    received_windows = []

    async def on_window(window):
        received_windows.append(window)
        pts_counts = [f.num_points for f in window]
        print(
            f"[Window #{len(received_windows)}] "
            f"프레임 수: {len(window)}, "
            f"각 프레임 포인트 수: {pts_counts}"
        )
        if len(received_windows) >= 3:
            capture.stop()

    capture = RadarCapture(
        window_size=16,
        stride=8,
        mock_mode=True,
    )

    print("=== RadarCapture Mock 테스트 시작 ===")
    print("16 프레임 윈도우 3회 수집 후 종료")

    asyncio.run(capture.capture_loop(on_window, mock_scenario_weights=[0.8, 0.15, 0.05]))

    print(f"\n총 {len(received_windows)}개 윈도우 수신 완료")
    print(f"마지막 윈도우 첫 프레임: {received_windows[-1][0]}")
