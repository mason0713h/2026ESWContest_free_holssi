"""
main.py
=======
Fall Guardian 메인 실행 파일.

전체 파이프라인을 asyncio 기반으로 통합하여 실행한다:

    레이더 캡처 (RadarCapture)
        ↓ 슬라이딩 윈도우 (16 프레임)
    포인트클라우드 전처리 (PointCloudPreprocessor)
        ↓ (T, N, 6) 텐서
    다중 인원 추적 (MultiPersonTracker)
        ↓ 인원별 포인트 분리
    TensorRT 추론 (TRTInferenceEngine) / PyTorch 폴백
        ↓ 낙상 확률
    Long-lie 판별 (LongLieDetector)
        ↓ 알림 레벨
    에스컬레이션 (AlertEscalationManager)
        → MQTT / SMS / 119 웹훅

실행:
    python main.py                    # 실제 하드웨어 모드
    python main.py --mock             # Mock 모드 (하드웨어 없음)
    python main.py --mock --debug     # Mock + 디버그 로그
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml

# ──────────────────────────────────────────────
# 프로젝트 루트를 sys.path에 추가
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from radar.capture import RadarCapture, PointCloud
from radar.preprocessor import PointCloudPreprocessor
from radar.multi_tracker import MultiPersonTracker
from model.fall_detector import FallDetector
from model.long_lie_detector import LongLieDetector, AlertLevel
from model.tensorrt_inference import TRTInferenceEngine
from alert.mqtt_publisher import MQTTAlertPublisher
from alert.voice_confirm import VoiceConfirmation
from alert.escalation import AlertEscalationManager, SMSSender, EmergencyWebhookSender

logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False, log_file: str = "logs/fall_guardian.log") -> None:
    """로깅 설정."""
    level = logging.DEBUG if debug else logging.INFO
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    handlers = [logging.StreamHandler(sys.stdout)]
    try:
        from logging.handlers import RotatingFileHandler
        handlers.append(
            RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
            )
        )
    except Exception:
        pass

    logging.basicConfig(level=level, format=log_format, handlers=handlers)
    logging.getLogger("paho").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """설정 파일 로드."""
    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning("설정 파일 없음 (%s). 기본값 사용.", config_path)
        return {}

    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class FallGuardianPipeline:
    """
    낙상 가디언 메인 파이프라인.

    모든 컴포넌트를 초기화하고 비동기 루프를 실행한다.

    Args:
        config: 설정 딕셔너리
        mock_mode: 하드웨어 없는 Mock 모드
    """

    def __init__(self, config: dict, mock_mode: bool = False) -> None:
        self.config = config
        self.mock_mode = mock_mode
        self._running = False
        self._stats = {
            "frames_processed": 0,
            "windows_processed": 0,
            "falls_detected": 0,
            "alerts_sent": 0,
            "start_time": 0.0,
        }

        # 설정값 추출 (기본값 포함)
        radar_cfg = config.get("radar", {})
        prep_cfg = config.get("preprocessing", {})
        model_cfg = config.get("model", {})
        tracking_cfg = config.get("tracking", {})
        mqtt_cfg = config.get("mqtt", {})
        alert_cfg = config.get("alert", {})
        long_lie_cfg = config.get("long_lie", {})

        # ── 컴포넌트 초기화 ─────────────────────────

        # 레이더 캡처
        self.capture = RadarCapture(
            config_port=radar_cfg.get("config_port", "/dev/ttyUSB0"),
            data_port=radar_cfg.get("data_port", "/dev/ttyUSB1"),
            config_baudrate=radar_cfg.get("config_baudrate", 115200),
            data_baudrate=radar_cfg.get("data_baudrate", 921600),
            window_size=radar_cfg.get("window_size", 16),
            stride=radar_cfg.get("stride", 8),
            mock_mode=mock_mode or radar_cfg.get("mock_mode", False),
            cfg_file=radar_cfg.get("cfg_file"),
        )

        # 전처리기
        self.preprocessor = PointCloudPreprocessor(
            target_points=prep_cfg.get("target_points", 64),
            dbscan_eps=prep_cfg.get("dbscan_eps", 0.3),
            dbscan_min_samples=prep_cfg.get("dbscan_min_samples", 3),
            x_range=tuple(prep_cfg.get("x_range", [-3.0, 3.0])),
            y_range=tuple(prep_cfg.get("y_range", [0.0, 5.0])),
            z_range=tuple(prep_cfg.get("z_range", [-0.5, 2.5])),
            vmax=prep_cfg.get("vmax", 3.0),
        )

        # 다중 인원 추적기
        self.tracker = MultiPersonTracker(
            max_persons=tracking_cfg.get("max_persons", 5),
            track_timeout=tracking_cfg.get("track_timeout_seconds", 5.0),
            min_track_frames=tracking_cfg.get("min_track_frames", 3),
        )

        # 추론 엔진 (TRT 우선, 없으면 PyTorch)
        self.trt_engine = TRTInferenceEngine(
            engine_path=model_cfg.get("trt_path", "model/weights/fall_guardian.trt"),
            fall_threshold=model_cfg.get("fall_threshold", 0.7),
        )
        self.trt_engine.load()

        # PyTorch 폴백 모델
        self.pt_model: Optional[FallDetector] = None
        self._init_pytorch_model(model_cfg)

        # Long-lie 감지기
        self.long_lie = LongLieDetector(
            z_floor_threshold=long_lie_cfg.get("z_floor_threshold", 0.3),
            level1_seconds=long_lie_cfg.get("level1_seconds", 30.0),
            level2_seconds=long_lie_cfg.get("level2_seconds", 60.0),
            level3_seconds=long_lie_cfg.get("level3_seconds", 120.0),
        )

        # MQTT 발행자
        self.mqtt_publisher = MQTTAlertPublisher(
            broker_host=mqtt_cfg.get("broker_host", "localhost"),
            broker_port=mqtt_cfg.get("broker_port", 1883),
            client_id=mqtt_cfg.get("client_id", "fall_guardian_001"),
            mock_mode=mock_mode,
        )

        # 음성 확인
        self.voice_confirm = VoiceConfirmation(
            response_timeout=alert_cfg.get("voice_response_timeout", 30.0),
            mock_mode=mock_mode,
        )

        # SMS 발송기
        self.sms_sender = SMSSender(
            from_number=alert_cfg.get("twilio_from_number", ""),
            to_number=alert_cfg.get("guardian_phone", ""),
            mock_mode=mock_mode,
        )

        # 119 웹훅
        self.webhook_sender = EmergencyWebhookSender(
            webhook_url=alert_cfg.get("emergency_webhook_url", ""),
            secret=alert_cfg.get("emergency_webhook_secret", ""),
            mock_mode=mock_mode,
        )

        # 에스컬레이션 관리자
        self.escalation = AlertEscalationManager(
            mqtt_publisher=self.mqtt_publisher,
            sms_sender=self.sms_sender,
            webhook_sender=self.webhook_sender,
            voice_confirmation=self.voice_confirm,
            level2_delay=alert_cfg.get("voice_response_timeout", 30.0),
            level3_delay=60.0,
            cooldown_seconds=alert_cfg.get("alert_cooldown_seconds", 300.0),
            mock_mode=mock_mode,
        )

        # Long-lie 콜백 등록
        self.long_lie.register_callback(
            AlertLevel.FALL, self._on_fall_level1
        )
        self.long_lie.register_callback(
            AlertLevel.LONG_LIE_30, self._on_long_lie
        )
        self.long_lie.register_callback(
            AlertLevel.LONG_LIE_60, self._on_long_lie
        )
        self.long_lie.register_callback(
            AlertLevel.LONG_LIE_120, self._on_long_lie_emergency
        )

        logger.info("FallGuardianPipeline 초기화 완료 (mock=%s)", mock_mode)

    def _init_pytorch_model(self, model_cfg: dict) -> None:
        """PyTorch 폴백 모델 초기화."""
        try:
            import torch
            self.pt_model = FallDetector(
                fall_threshold=model_cfg.get("fall_threshold", 0.7),
            )
            ckpt_path = model_cfg.get("checkpoint_path", "")
            if ckpt_path and Path(ckpt_path).exists():
                self.pt_model.load(ckpt_path)
                logger.info("PyTorch 모델 로드: %s", ckpt_path)
            else:
                logger.info("PyTorch 폴백 모델 랜덤 가중치 사용 (미학습)")
            self.pt_model.eval()
        except Exception as e:
            logger.warning("PyTorch 모델 초기화 실패: %s", e)
            self.pt_model = None

    # ── 낙상 감지 추론 ───────────────────────────

    def _infer(self, tensor: np.ndarray) -> tuple[bool, float]:
        """
        추론 실행 (TRT → PyTorch → Fallback 순).

        Args:
            tensor: (T, N, C) 전처리된 포인트 텐서

        Returns:
            (fall_detected, fall_probability)
        """
        # TRT 추론 시도
        fall, prob, lat = self.trt_engine.infer(tensor)
        if lat < 1000:  # 유효한 응답
            return fall, prob

        # PyTorch 폴백
        if self.pt_model is not None:
            try:
                import torch
                x = torch.from_numpy(tensor).unsqueeze(0)  # (1, T, N, C)
                fall_det, confidence = self.pt_model.predict(x[0])
                return fall_det, confidence
            except Exception as e:
                logger.debug("PyTorch 추론 오류: %s", e)

        return False, 0.0

    # ── Long-lie 콜백 ─────────────────────────

    def _on_fall_level1(self, person_id: int, state) -> None:
        """낙상 Level 1 콜백 (동기) - 에스컬레이션 태스크 예약."""
        asyncio.create_task(
            self.escalation.trigger(
                person_id=person_id,
                confidence=state.alert_level,
                position=None,
            )
        )

    def _on_long_lie(self, person_id: int, state) -> None:
        """장시간 쓰러짐 콜백."""
        if self.mqtt_publisher:
            self.mqtt_publisher.publish_long_lie(
                person_id=person_id,
                fall_duration=state.fall_duration,
                alert_level=int(state.alert_level),
            )

    def _on_long_lie_emergency(self, person_id: int, state) -> None:
        """120초 긴급 콜백."""
        logger.critical(
            "긴급! 인원 #%d 장시간 쓰러짐 (%.0f초) - 119 호출 필요!",
            person_id, state.fall_duration,
        )
        self._on_long_lie(person_id, state)

    # ── 메인 처리 콜백 ────────────────────────

    async def _on_window(self, window: List[PointCloud]) -> None:
        """
        슬라이딩 윈도우 수신 시 처리 콜백.

        전처리 → 추적 → 추론 → Long-lie → 알림
        """
        self._stats["windows_processed"] += 1

        # 1. 전체 윈도우 전처리
        try:
            tensor = self.preprocessor.process_window(window)
            # tensor: (T, N, 6)
        except Exception as e:
            logger.error("전처리 오류: %s", e)
            return

        # 2. 최신 프레임으로 다중 인원 추적 갱신
        latest_pc = window[-1]
        if len(latest_pc.points) > 0:
            active_persons = self.tracker.update(latest_pc.points)
        else:
            active_persons = []

        # 3. 인원별 독립 추론
        if not active_persons:
            # 인원 없음 → 전체 씬에 대해 추론
            fall_detected, fall_prob = self._infer(tensor)
            if fall_detected:
                logger.warning(
                    "낙상 감지 (추적 없음): confidence=%.3f", fall_prob
                )
        else:
            for person in active_persons:
                # 인원별로 DBSCAN 분리된 포인트 히스토리를 독립적으로 전처리
                history = self.tracker.get_person_window(person.person_id) or []
                person_tensor = self.preprocessor.process_person_history(
                    history, window_size=tensor.shape[0]
                )
                fall_detected, fall_prob = self._infer(person_tensor)

                # 오경보 억제: 연속 3 프레임 낙상 감지 시만 인정
                self.tracker.update_fall_status(
                    person.person_id, fall_detected, fall_prob
                )

                # 4. Long-lie 업데이트 (해당 인원의 최신 분리 포인트 사용)
                person_pts = history[-1] if history else np.zeros((0, 6), dtype=np.float32)
                alert_level = self.long_lie.update(
                    person_id=person.person_id,
                    fall_detected=fall_detected and fall_prob >= 0.7,
                    fall_confidence=fall_prob,
                    person_points=person_pts,
                )

                if fall_detected and fall_prob >= 0.7:
                    self._stats["falls_detected"] += 1
                    logger.warning(
                        "낙상 감지: person_id=%d, prob=%.3f, level=%s",
                        person.person_id, fall_prob, alert_level.name,
                    )

    # ── 파이프라인 실행 ───────────────────────

    async def run(self) -> None:
        """파이프라인 메인 루프 실행."""
        self._running = True
        self._stats["start_time"] = time.time()

        # 레이더 시리얼 포트 연결 및 .cfg 전송 (mock 모드에서는 내부에서 스킵됨)
        self.capture.connect()

        # MQTT 연결
        if not self.mqtt_publisher.connect():
            logger.warning("MQTT 연결 실패. 알림 발송 불가.")

        logger.info("Fall Guardian 파이프라인 시작 (mock=%s)", self.mock_mode)

        # Heartbeat 태스크
        heartbeat_task = asyncio.create_task(
            self.mqtt_publisher.publish_heartbeat_loop(interval=30.0)
        )

        # 레이더 캡처 루프
        try:
            await self.capture.capture_loop(
                callback=self._on_window,
                mock_scenario_weights=[0.85, 0.10, 0.05],
            )
        except asyncio.CancelledError:
            logger.info("캡처 루프 취소됨")
        finally:
            heartbeat_task.cancel()
            self.capture.stop()
            self.capture.disconnect()
            self.mqtt_publisher.disconnect()
            self._print_stats()

    def stop(self) -> None:
        """파이프라인 중지."""
        self._running = False
        self.capture.stop()
        logger.info("Fall Guardian 중지")

    def _print_stats(self) -> None:
        """실행 통계 출력."""
        elapsed = time.time() - self._stats["start_time"]
        logger.info(
            "=== 실행 통계 ===\n"
            "  실행 시간: %.1f초\n"
            "  처리 윈도우: %d\n"
            "  낙상 감지: %d회\n"
            "  MQTT 발행: %d회",
            elapsed,
            self._stats["windows_processed"],
            self._stats["falls_detected"],
            self.mqtt_publisher.publish_count,
        )


# ──────────────────────────────────────────────
# 메인 엔트리포인트
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """커맨드라인 인자 파싱."""
    parser = argparse.ArgumentParser(
        description="Fall Guardian - 4D mmWave 레이더 낙상 감지 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python main.py --mock             # Mock 모드로 실행 (하드웨어 없음)
  python main.py --mock --debug     # Mock + 디버그 로그
  python main.py --config config/config.yaml  # 설정 파일 지정
        """,
    )
    parser.add_argument("--mock", action="store_true", help="Mock 모드 (하드웨어 없이 실행)")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="설정 파일 경로")
    parser.add_argument("--debug", action="store_true", help="디버그 로그 활성화")
    parser.add_argument("--log_file", type=str, default="logs/fall_guardian.log")
    return parser.parse_args()


async def main() -> None:
    """비동기 메인 함수."""
    args = parse_args()
    setup_logging(debug=args.debug, log_file=args.log_file)
    config = load_config(args.config)

    pipeline = FallGuardianPipeline(config=config, mock_mode=args.mock)

    # SIGINT/SIGTERM 핸들러
    loop = asyncio.get_event_loop()

    def shutdown_handler():
        logger.info("종료 신호 수신. 안전하게 종료 중...")
        # pipeline.stop()은 _running 플래그만 내리고, 실제 종료는 run() 내부의
        # 캡처 루프가 다음 반복에서 이를 감지하고 빠져나오며 완료된다.
        # loop.stop()을 여기서 동기 호출하면 run() 코루틴이 아직 완전히
        # unwind되지 않은 상태에서 루프가 멈춰 "Event loop stopped before
        # Future completed" RuntimeError가 발생하므로 호출하지 않는다.
        pipeline.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_handler)

    logger.info("=" * 60)
    logger.info("  Fall Guardian 시작")
    logger.info("  모드: %s", "Mock" if args.mock else "실제 하드웨어")
    logger.info("  설정: %s", args.config)
    logger.info("=" * 60)

    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())
