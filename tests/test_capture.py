"""
tests/test_capture.py
======================
RadarCapture / TLV 파싱 단위 테스트.

TI mmWave SDK 프레임 헤더 + Enhanced Point Cloud(TLV 1006) 프로토콜을
합성 바이트 패킷으로 직접 구성해 파싱 로직을 검증한다. 실제 레이더
하드웨어 없이도 UART/TLV 디코딩 정확성을 확인할 수 있다.

테스트 항목:
  - MAGIC_WORD 탐색
  - 프레임 헤더 파싱 (정상/손상/길이 부족)
  - TLV 헤더 파싱
  - Enhanced Point Cloud 포인트 레코드 디코딩
  - parse_packet() 엔드투엔드 (헤더 + TLV 전체)
  - FrameWindow 슬라이딩 윈도우 (window_size/stride)
  - PointCloud 속성 (xyz/velocity/snr)
  - Mock 프레임 생성 (normal/fall/empty 시나리오)
  - .cfg 파일 로드 vs 내장 기본 명령어 폴백
  - capture_loop (mock 모드) 비동기 통합
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from radar.capture import (
    FRAME_HEADER_SIZE,
    MAGIC_WORD,
    TLV_HEADER_SIZE,
    TLV_TYPE_ENHANCED_POINT_CLOUD,
    FrameWindow,
    PointCloud,
    RadarCapture,
)


def make_point_record(x, y, z, vel, snr_raw, noise_raw) -> bytes:
    """Enhanced Point Cloud 포인트 레코드(20 bytes) 1개 생성."""
    return struct.pack("<ffffHH", x, y, z, vel, snr_raw, noise_raw)


def make_tlv(tlv_type: int, payload: bytes) -> bytes:
    """TLV 헤더(8 bytes) + payload."""
    return struct.pack("<II", tlv_type, len(payload)) + payload


def make_frame_packet(frame_number: int, tlvs: list[bytes]) -> bytes:
    """프레임 헤더(40 bytes) + TLV 목록을 이어붙인 완전한 패킷 생성."""
    tlv_bytes = b"".join(tlvs)
    total_len = FRAME_HEADER_SIZE + len(tlv_bytes)
    header = MAGIC_WORD + struct.pack(
        "<IIIIIIII",
        1,              # version
        total_len,      # total_packet_len
        0,              # platform
        frame_number,   # frame_number
        0,              # cpu_cycles
        0,              # num_objects
        len(tlvs),      # num_tlvs
        0,              # subframe_idx
    )
    return header + tlv_bytes


@pytest.fixture
def capture():
    return RadarCapture(mock_mode=True, window_size=4, stride=2)


class TestMagicWordSearch:
    def test_finds_magic_word_at_start(self, capture):
        data = bytearray(MAGIC_WORD + b"\x00" * 10)
        assert capture._find_magic_word(data) == 0

    def test_finds_magic_word_with_leading_garbage(self, capture):
        data = bytearray(b"\xff\xff\xff" + MAGIC_WORD + b"\x00" * 4)
        assert capture._find_magic_word(data) == 3

    def test_returns_minus_one_when_absent(self, capture):
        data = bytearray(b"\x00" * 20)
        assert capture._find_magic_word(data) == -1


class TestFrameHeaderParsing:
    def test_parses_valid_header(self, capture):
        packet = make_frame_packet(frame_number=42, tlvs=[])
        header = capture._parse_frame_header(packet)
        assert header is not None
        assert header["frame_number"] == 42
        assert header["num_tlvs"] == 0
        assert header["total_len"] == FRAME_HEADER_SIZE

    def test_rejects_wrong_magic_word(self, capture):
        bad = b"\x00" * 8 + b"\x00" * 32
        assert capture._parse_frame_header(bad) is None

    def test_rejects_too_short_data(self, capture):
        assert capture._parse_frame_header(MAGIC_WORD) is None


class TestTLVHeaderParsing:
    def test_parses_valid_tlv_header(self, capture):
        data = struct.pack("<II", TLV_TYPE_ENHANCED_POINT_CLOUD, 20)
        result = capture._parse_tlv_header(data, 0)
        assert result == (TLV_TYPE_ENHANCED_POINT_CLOUD, 20)

    def test_returns_none_when_insufficient_bytes(self, capture):
        data = b"\x00\x00\x00"
        assert capture._parse_tlv_header(data, 0) is None


class TestEnhancedPointCloudParsing:
    def test_decodes_single_point_record(self, capture):
        record = make_point_record(1.5, 2.5, 0.5, -0.3, 200, 30)
        points = capture._parse_enhanced_point_cloud(record, 0, len(record))
        assert points.shape == (1, 6)
        x, y, z, vel, snr, noise = points[0]
        assert x == pytest.approx(1.5)
        assert y == pytest.approx(2.5)
        assert z == pytest.approx(0.5)
        assert vel == pytest.approx(-0.3)
        assert snr == pytest.approx(20.0)   # snr_raw(200) * 0.1
        assert noise == pytest.approx(3.0)  # noise_raw(30) * 0.1

    def test_decodes_multiple_point_records(self, capture):
        records = b"".join(
            make_point_record(float(i), 0.0, 0.0, 0.0, 100, 10) for i in range(5)
        )
        points = capture._parse_enhanced_point_cloud(records, 0, len(records))
        assert points.shape == (5, 6)
        assert list(points[:, 0]) == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])

    def test_zero_length_returns_empty_array(self, capture):
        points = capture._parse_enhanced_point_cloud(b"", 0, 0)
        assert points.shape == (0, 6)


class TestParsePacketEndToEnd:
    def test_parses_packet_with_point_cloud_tlv(self, capture):
        records = b"".join(
            make_point_record(float(i), 1.0, 1.0, 0.0, 150, 20) for i in range(3)
        )
        tlv = make_tlv(TLV_TYPE_ENHANCED_POINT_CLOUD, records)
        packet = make_frame_packet(frame_number=7, tlvs=[tlv])

        pc = capture.parse_packet(packet)
        assert pc is not None
        assert pc.frame_number == 7
        assert pc.num_points == 3
        assert pc.xyz.shape == (3, 3)

    def test_parses_packet_with_no_tlvs(self, capture):
        packet = make_frame_packet(frame_number=1, tlvs=[])
        pc = capture.parse_packet(packet)
        assert pc is not None
        assert pc.num_points == 0

    def test_ignores_unknown_tlv_type(self, capture):
        unknown_tlv = make_tlv(9999, b"\x00" * 16)
        packet = make_frame_packet(frame_number=2, tlvs=[unknown_tlv])
        pc = capture.parse_packet(packet)
        assert pc is not None
        assert pc.num_points == 0

    def test_returns_none_for_corrupted_header(self, capture):
        assert capture.parse_packet(b"\x00" * 50) is None

    def test_multiple_tlvs_combined(self, capture):
        rec_a = make_point_record(0.0, 0.0, 0.0, 0.0, 100, 10)
        rec_b = make_point_record(1.0, 1.0, 1.0, 0.0, 100, 10)
        tlv1 = make_tlv(TLV_TYPE_ENHANCED_POINT_CLOUD, rec_a)
        tlv2 = make_tlv(TLV_TYPE_ENHANCED_POINT_CLOUD, rec_b)
        packet = make_frame_packet(frame_number=3, tlvs=[tlv1, tlv2])

        pc = capture.parse_packet(packet)
        assert pc.num_points == 2


class TestPointCloudProperties:
    def test_num_points_matches_array_length(self):
        pts = np.zeros((5, 6), dtype=np.float32)
        pc = PointCloud(timestamp=0.0, frame_number=0, points=pts)
        assert pc.num_points == 5

    def test_xyz_velocity_snr_slices(self):
        pts = np.array(
            [[1.0, 2.0, 3.0, 0.5, 20.0, 1.0]],
            dtype=np.float32,
        )
        pc = PointCloud(timestamp=0.0, frame_number=0, points=pts)
        assert pc.xyz.tolist() == [[1.0, 2.0, 3.0]]
        assert pc.velocity.tolist() == [0.5]
        assert pc.snr.tolist() == [20.0]


class TestFrameWindowSliding:
    def test_no_window_until_full(self):
        fw = FrameWindow(window_size=3, stride=1)
        pc = PointCloud(timestamp=0.0, frame_number=0, points=np.zeros((0, 6)))
        assert fw.push(pc) is None
        assert fw.push(pc) is None

    def test_returns_window_once_full(self):
        fw = FrameWindow(window_size=3, stride=1)
        pc = PointCloud(timestamp=0.0, frame_number=0, points=np.zeros((0, 6)))
        fw.push(pc)
        fw.push(pc)
        window = fw.push(pc)
        assert window is not None
        assert len(window) == 3

    def test_stride_skips_intermediate_frames(self):
        fw = FrameWindow(window_size=2, stride=2)
        pc = PointCloud(timestamp=0.0, frame_number=0, points=np.zeros((0, 6)))
        assert fw.push(pc) is None       # frame 1 -> 윈도우 미충족
        assert fw.push(pc) is not None   # frame 2 -> 윈도우 충족
        assert fw.push(pc) is None       # frame 3 -> stride 미충족
        assert fw.push(pc) is not None   # frame 4 -> stride 충족

    def test_window_drops_oldest_frame(self):
        fw = FrameWindow(window_size=2, stride=1)
        pc1 = PointCloud(timestamp=1.0, frame_number=1, points=np.zeros((0, 6)))
        pc2 = PointCloud(timestamp=2.0, frame_number=2, points=np.zeros((0, 6)))
        pc3 = PointCloud(timestamp=3.0, frame_number=3, points=np.zeros((0, 6)))
        fw.push(pc1)
        fw.push(pc2)
        window = fw.push(pc3)
        assert [f.frame_number for f in window] == [2, 3]


class TestMockFrameGeneration:
    def test_normal_scenario_has_points_above_floor(self, capture):
        pc = capture._generate_mock_frame("normal")
        assert pc.num_points > 0
        assert pc.xyz[:, 2].mean() > 0.3  # 서있는 높이대

    def test_fall_scenario_has_points_near_floor(self, capture):
        pc = capture._generate_mock_frame("fall")
        assert pc.num_points > 0
        assert pc.xyz[:, 2].mean() < 0.5  # 바닥 근처

    def test_empty_scenario_has_no_points(self, capture):
        pc = capture._generate_mock_frame("empty")
        assert pc.num_points == 0

    def test_frame_number_increments(self, capture):
        pc1 = capture._generate_mock_frame("normal")
        pc2 = capture._generate_mock_frame("normal")
        assert pc2.frame_number == pc1.frame_number + 1


class TestCfgCommandLoading:
    def test_falls_back_to_builtin_commands_when_no_cfg_file(self, capture):
        commands = capture._load_cfg_commands()
        assert any("sensorStart" in c for c in commands)
        assert any("frameCfg" in c for c in commands)

    def test_loads_real_cfg_file(self, tmp_path):
        cfg = tmp_path / "test.cfg"
        cfg.write_text(
            "% comment line\n"
            "sensorStop\n"
            "\n"
            "frameCfg 0 2 16 0 100 1 0\n"
        )
        capture = RadarCapture(mock_mode=True, cfg_file=str(cfg))
        commands = capture._load_cfg_commands()
        assert commands == ["sensorStop\n", "frameCfg 0 2 16 0 100 1 0\n"]

    def test_raises_when_cfg_file_missing(self):
        capture = RadarCapture(mock_mode=True, cfg_file="/nonexistent/path.cfg")
        with pytest.raises(FileNotFoundError):
            capture._load_cfg_commands()


class TestConnectDisconnectMockMode:
    def test_connect_is_noop_in_mock_mode(self, capture):
        capture.connect()  # 예외 없이 통과해야 함 (시리얼 포트 열지 않음)
        assert capture._ser_config is None
        assert capture._ser_data is None

    def test_disconnect_is_safe_without_connect(self, capture):
        capture.disconnect()  # 연결 없이 호출해도 예외 없어야 함


class TestCaptureLoopMockIntegration:
    @pytest.mark.asyncio
    async def test_capture_loop_emits_windows(self):
        capture = RadarCapture(mock_mode=True, window_size=2, stride=2)
        windows = []

        async def on_window(window):
            windows.append(window)
            if len(windows) >= 2:
                capture.stop()

        await capture.capture_loop(on_window, mock_scenario_weights=[1.0, 0.0, 0.0])

        assert len(windows) == 2
        assert all(len(w) == 2 for w in windows)

    def test_stop_clears_running_flag(self, capture):
        capture._running = True
        capture.stop()
        assert capture._running is False
