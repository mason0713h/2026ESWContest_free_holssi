"""
alert/escalation.py
===================
낙상 알림 단계적 에스컬레이션 로직.

에스컬레이션 단계:
    Level 1 (감지 즉시):   MQTT로 보호자 앱 알림
    Level 2 (30초 무응답): SMS 발송 (Twilio API)
    Level 3 (60초 무응답 또는 long-lie): 119 게이트웨이 HTTP 웹훅 발송

각 단계는 독립적으로 실행되며, 중복 발송을 방지하기 위해
쿨다운 타이머와 발송 이력을 관리한다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, List, Optional

import aiohttp  # type: ignore  (없으면 폴백 처리)

logger = logging.getLogger(__name__)

# Twilio 가용성 확인
try:
    from twilio.rest import Client as TwilioClient  # type: ignore
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    logger.warning("twilio 패키지 미설치. SMS 발송은 Mock 모드로 동작합니다.")


class EscalationLevel(IntEnum):
    """에스컬레이션 단계."""
    NONE = 0
    GUARDIAN_APP = 1    # 보호자 앱 MQTT 알림
    SMS = 2             # SMS 발송
    EMERGENCY_119 = 3   # 119 웹훅


@dataclass
class EscalationEvent:
    """에스컬레이션 이벤트 기록."""
    person_id: int
    level: EscalationLevel
    timestamp: float
    method: str
    success: bool
    response_code: Optional[int] = None
    error_msg: Optional[str] = None


class SMSSender:
    """
    SMS 발송기 (Twilio API).

    Twilio API를 사용하여 보호자에게 SMS를 발송한다.
    환경변수 TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN으로 인증.

    Args:
        account_sid: Twilio 계정 SID (None이면 환경변수 사용)
        auth_token: Twilio 인증 토큰
        from_number: 발신 전화번호
        to_number: 수신 전화번호
        mock_mode: True면 실제 발송 없이 로그만 출력
    """

    def __init__(
        self,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
        from_number: str = "",
        to_number: str = "",
        mock_mode: bool = False,
    ) -> None:
        self.from_number = from_number
        self.to_number = to_number
        self.mock_mode = mock_mode or not TWILIO_AVAILABLE

        sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID", "")
        token = auth_token or os.getenv("TWILIO_AUTH_TOKEN", "")

        if not self.mock_mode and sid and token and TWILIO_AVAILABLE:
            try:
                self._client = TwilioClient(sid, token)
                logger.info("Twilio SMS 클라이언트 초기화 완료")
            except Exception as e:
                logger.warning("Twilio 초기화 실패: %s. Mock 모드 전환", e)
                self._client = None
                self.mock_mode = True
        else:
            self._client = None
            if not self.mock_mode:
                logger.warning("Twilio 자격증명 없음. Mock 모드 사용")
                self.mock_mode = True

    def send(self, message: str, to: Optional[str] = None) -> bool:
        """
        SMS 발송.

        Args:
            message: 발송할 메시지 내용
            to: 수신 번호 (None이면 self.to_number 사용)

        Returns:
            발송 성공 여부
        """
        recipient = to or self.to_number

        if self.mock_mode:
            logger.info("[Mock SMS] → %s: %s", recipient, message)
            return True

        if not self._client:
            logger.error("Twilio 클라이언트 미초기화")
            return False

        try:
            msg = self._client.messages.create(
                body=message,
                from_=self.from_number,
                to=recipient,
            )
            logger.info("SMS 발송 성공: SID=%s, to=%s", msg.sid, recipient)
            return True
        except Exception as e:
            logger.error("SMS 발송 실패: %s", e)
            return False


class EmergencyWebhookSender:
    """
    119 게이트웨이 HTTP 웹훅 발송기 (개념실증).

    실제 배포 시 행정안전부/소방청 공개 API 또는
    응급의료 정보 시스템(EMIS) 연계로 대체한다.

    Args:
        webhook_url: 웹훅 엔드포인트 URL
        secret: 인증 시크릿 (HTTP 헤더로 전송)
        timeout: 요청 타임아웃 (초)
        mock_mode: True면 실제 HTTP 요청 없이 시뮬레이션
    """

    def __init__(
        self,
        webhook_url: str = "",
        secret: str = "",
        timeout: float = 10.0,
        mock_mode: bool = False,
    ) -> None:
        self.webhook_url = webhook_url
        self.secret = secret
        self.timeout = timeout
        self.mock_mode = mock_mode or not webhook_url

    async def send_async(self, payload: dict) -> bool:
        """
        비동기 HTTP 웹훅 발송.

        Args:
            payload: 전송할 JSON 데이터

        Returns:
            발송 성공 여부
        """
        if self.mock_mode:
            logger.warning("[Mock 119 웹훅] %s", json.dumps(payload, ensure_ascii=False))
            return True

        headers = {
            "Content-Type": "application/json",
            "X-Guardian-Secret": self.secret,
            "X-Guardian-Version": "1.0",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    if resp.status in (200, 201, 202):
                        logger.warning(
                            "119 웹훅 발송 성공: status=%d", resp.status
                        )
                        return True
                    else:
                        text = await resp.text()
                        logger.error(
                            "119 웹훅 발송 실패: status=%d, body=%s",
                            resp.status, text[:200],
                        )
                        return False
        except Exception as e:
            logger.error("119 웹훅 오류: %s", e)
            return False


class AlertEscalationManager:
    """
    낙상 알림 에스컬레이션 관리자.

    감지 이벤트를 받아 설정된 단계에 따라 순차적으로 알림을 발송한다.
    각 인원별 에스컬레이션 상태와 쿨다운을 관리한다.

    Args:
        mqtt_publisher: MQTT 발행자
        sms_sender: SMS 발송기
        webhook_sender: 119 웹훅 발송기
        voice_confirmation: 음성 확인 모듈
        level2_delay: Level 1 → Level 2 지연 (초, 기본 30)
        level3_delay: Level 1 → Level 3 지연 (초, 기본 60)
        cooldown_seconds: 동일 인원 중복 에스컬레이션 방지 쿨다운 (초)
        mock_mode: True면 모든 실제 발송 비활성화
    """

    def __init__(
        self,
        mqtt_publisher=None,
        sms_sender: Optional[SMSSender] = None,
        webhook_sender: Optional[EmergencyWebhookSender] = None,
        voice_confirmation=None,
        level2_delay: float = 30.0,
        level3_delay: float = 60.0,
        cooldown_seconds: float = 300.0,
        mock_mode: bool = False,
    ) -> None:
        self.mqtt_publisher = mqtt_publisher
        self.sms_sender = sms_sender or SMSSender(mock_mode=True)
        self.webhook_sender = webhook_sender or EmergencyWebhookSender(mock_mode=True)
        self.voice_confirmation = voice_confirmation
        self.level2_delay = level2_delay
        self.level3_delay = level3_delay
        self.cooldown_seconds = cooldown_seconds
        self.mock_mode = mock_mode

        # 인원별 에스컬레이션 상태
        self._escalation_tasks: Dict[int, asyncio.Task] = {}
        self._last_escalation: Dict[int, float] = {}
        self._event_history: List[EscalationEvent] = []

        # 에스컬레이션 취소 플래그 (음성 응답 시)
        self._cancelled: Dict[int, bool] = {}

        logger.info(
            "AlertEscalationManager 초기화: L2=%.0fs, L3=%.0fs, cooldown=%.0fs",
            level2_delay, level3_delay, cooldown_seconds,
        )

    def _record_event(
        self,
        person_id: int,
        level: EscalationLevel,
        method: str,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """이벤트 기록."""
        event = EscalationEvent(
            person_id=person_id,
            level=level,
            timestamp=time.time(),
            method=method,
            success=success,
            error_msg=error,
        )
        self._event_history.append(event)
        if len(self._event_history) > 1000:
            self._event_history = self._event_history[-500:]

    def _is_in_cooldown(self, person_id: int) -> bool:
        """쿨다운 중인지 확인."""
        last = self._last_escalation.get(person_id, 0.0)
        return time.time() - last < self.cooldown_seconds

    async def _escalation_pipeline(
        self,
        person_id: int,
        confidence: float,
        position: Optional[list],
    ) -> None:
        """
        에스컬레이션 파이프라인 (비동기 코루틴).

        Level 1 → 음성 확인 (30초) → Level 2 → Level 3 순으로 진행.
        음성으로 "괜찮아" 응답 시 즉시 중단.
        """
        self._cancelled[person_id] = False
        self._last_escalation[person_id] = time.time()

        # ─── Level 1: 보호자 앱 MQTT ──────────────
        logger.warning("[Esc L1] 인원 #%d 보호자 앱 알림", person_id)
        if self.mqtt_publisher:
            success = self.mqtt_publisher.publish_fall_detected(
                person_id=person_id,
                confidence=confidence,
                position=position,
            )
        else:
            success = True
            logger.info("[Mock L1] MQTT 알림 발송 시뮬레이션")

        self._record_event(person_id, EscalationLevel.GUARDIAN_APP, "mqtt", success)

        # 음성 확인 시작 (비동기)
        voice_result = "timeout"
        if self.voice_confirmation:
            voice_result = await self.voice_confirmation.confirm_async(
                person_id=person_id,
                on_ok=self._on_voice_ok,
                on_escalate=None,
            )
        else:
            # Mock: 타임아웃 시뮬레이션
            logger.info("[Mock Voice] %.0f초 응답 대기...", self.level2_delay)
            await asyncio.sleep(self.level2_delay)

        if self._cancelled.get(person_id, False):
            logger.info("인원 #%d 에스컬레이션 취소 (음성 응답)", person_id)
            return

        if voice_result == "ok":
            logger.info("인원 #%d 자력 확인. 에스컬레이션 중단.", person_id)
            if self.mqtt_publisher:
                self.mqtt_publisher.publish_all_clear(person_id)
            return

        # ─── Level 2: SMS ──────────────────────────
        logger.warning("[Esc L2] 인원 #%d SMS 발송", person_id)
        sms_text = (
            f"[낙상 가디언 긴급 알림]\n"
            f"낙상이 감지되었습니다.\n"
            f"인원 ID: {person_id}\n"
            f"신뢰도: {confidence:.1%}\n"
            f"감지 시각: {time.strftime('%H:%M:%S')}\n"
            f"30초간 응답 없음. 확인이 필요합니다."
        )
        sms_success = self.sms_sender.send(sms_text)
        self._record_event(person_id, EscalationLevel.SMS, "sms", sms_success)

        if self._cancelled.get(person_id, False):
            return

        # Level 3까지 추가 대기
        remaining_delay = max(0.0, self.level3_delay - self.level2_delay)
        await asyncio.sleep(remaining_delay)

        if self._cancelled.get(person_id, False):
            return

        # ─── Level 3: 119 웹훅 ─────────────────────
        logger.warning("[Esc L3] 인원 #%d 119 웹훅 발송!", person_id)
        emergency_payload = {
            "event_type": "emergency_fall",
            "device_id": "fall_guardian_001",
            "person_id": person_id,
            "confidence": confidence,
            "position": position or [0.0, 0.0, 0.0],
            "timestamp": time.time(),
            "no_response_duration_seconds": self.level3_delay,
            "message": "낙상 후 장시간 무응답으로 응급 출동이 필요합니다.",
        }
        webhook_success = await self.webhook_sender.send_async(emergency_payload)
        self._record_event(
            person_id, EscalationLevel.EMERGENCY_119, "http_webhook", webhook_success
        )

        if self.mqtt_publisher:
            self.mqtt_publisher.publish_fall_confirmed(
                person_id=person_id,
                confidence=confidence,
                position=position,
            )

    async def _on_voice_ok(self, person_id: int) -> None:
        """음성 긍정 응답 시 에스컬레이션 취소."""
        self._cancelled[person_id] = True
        logger.info("인원 #%d 음성 응답으로 에스컬레이션 취소", person_id)
        if self.mqtt_publisher:
            self.mqtt_publisher.publish_all_clear(person_id)

    async def trigger(
        self,
        person_id: int,
        confidence: float,
        position: Optional[list] = None,
    ) -> None:
        """
        에스컬레이션 트리거.

        기존 에스컬레이션이 진행 중이면 무시하고,
        쿨다운 중이면 스킵한다.

        Args:
            person_id: 대상 인원 ID
            confidence: 낙상 신뢰도
            position: 위치 [x, y, z]
        """
        # 쿨다운 체크
        if self._is_in_cooldown(person_id):
            logger.debug("인원 #%d 쿨다운 중 - 에스컬레이션 스킵", person_id)
            return

        # 기존 태스크가 실행 중이면 취소
        if person_id in self._escalation_tasks:
            old_task = self._escalation_tasks[person_id]
            if not old_task.done():
                logger.info("인원 #%d 기존 에스컬레이션 취소 후 재시작", person_id)
                old_task.cancel()
                try:
                    await asyncio.wait_for(old_task, timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

        # 새 에스컬레이션 태스크 시작
        task = asyncio.create_task(
            self._escalation_pipeline(person_id, confidence, position)
        )
        self._escalation_tasks[person_id] = task
        logger.warning(
            "에스컬레이션 시작: person_id=%d, confidence=%.3f",
            person_id, confidence,
        )

    async def cancel(self, person_id: int) -> None:
        """특정 인원의 에스컬레이션 취소 (자력 회복 시)."""
        self._cancelled[person_id] = True
        task = self._escalation_tasks.get(person_id)
        if task and not task.done():
            task.cancel()
            logger.info("인원 #%d 에스컬레이션 취소", person_id)

    def get_history(self, person_id: Optional[int] = None) -> List[EscalationEvent]:
        """이벤트 히스토리 반환."""
        if person_id is None:
            return self._event_history.copy()
        return [e for e in self._event_history if e.person_id == person_id]


# ──────────────────────────────────────────────
# 단독 실행 테스트
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=== AlertEscalationManager 테스트 ===\n")

    # Mock 컴포넌트로 초기화
    sms = SMSSender(mock_mode=True)
    webhook = EmergencyWebhookSender(mock_mode=True)

    manager = AlertEscalationManager(
        mqtt_publisher=None,
        sms_sender=sms,
        webhook_sender=webhook,
        voice_confirmation=None,
        level2_delay=3.0,    # 테스트용: 3초
        level3_delay=6.0,    # 테스트용: 6초
        cooldown_seconds=10.0,
        mock_mode=True,
    )

    async def test_escalation():
        print("에스컬레이션 트리거...")
        await manager.trigger(
            person_id=1,
            confidence=0.92,
            position=[0.5, 2.3, 0.1],
        )

        # 파이프라인 완료 대기
        await asyncio.sleep(8.0)

        history = manager.get_history(person_id=1)
        print(f"\n이벤트 히스토리 ({len(history)}건):")
        for event in history:
            ts = time.strftime("%H:%M:%S", time.localtime(event.timestamp))
            print(
                f"  [{ts}] Level={event.level.name}, "
                f"method={event.method}, success={event.success}"
            )

    asyncio.run(test_escalation())
    print("\n에스컬레이션 테스트 완료!")
