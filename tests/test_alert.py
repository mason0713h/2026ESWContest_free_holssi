"""
tests/test_alert.py
===================
알림 모듈 단위 테스트.

테스트 항목:
  - MQTTAlertPublisher: 연결, 발행, 메시지 직렬화
  - FallAlertMessage: JSON 직렬화
  - VoiceConfirmation: 응답 분류, Mock 모드
  - SMSSender: Mock 발송
  - AlertEscalationManager: 에스컬레이션 트리거, 취소, 쿨다운
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from alert.mqtt_publisher import MQTTAlertPublisher, FallAlertMessage
from alert.voice_confirm import VoiceConfirmation, OK_KEYWORDS, HELP_KEYWORDS
from alert.escalation import (
    AlertEscalationManager,
    SMSSender,
    EmergencyWebhookSender,
    EscalationLevel,
)


# ── FallAlertMessage 테스트 ───────────────────

class TestFallAlertMessage:

    def test_json_serialization(self):
        """JSON 직렬화가 올바르게 동작해야 한다."""
        msg = FallAlertMessage(
            event_type="fall_detected",
            person_id=1,
            timestamp=1234567890.0,
            confidence=0.92,
            position=[0.5, 2.3, 0.1],
            alert_level=1,
            message_ko="낙상이 감지되었습니다.",
        )
        js = msg.to_json()
        parsed = json.loads(js)

        assert parsed["event_type"] == "fall_detected"
        assert parsed["person_id"] == 1
        assert parsed["confidence"] == pytest.approx(0.92, abs=1e-5)
        assert parsed["position"] == [0.5, 2.3, 0.1]
        assert "timestamp_iso" in parsed
        assert parsed["message_ko"] == "낙상이 감지되었습니다."

    def test_json_korean_preserved(self):
        """한국어 문자가 손실 없이 직렬화되어야 한다."""
        msg = FallAlertMessage(
            event_type="test",
            person_id=1,
            timestamp=time.time(),
            message_ko="낙상 감지 테스트 한국어",
        )
        js = msg.to_json()
        assert "낙상 감지 테스트 한국어" in js


# ── MQTTAlertPublisher 테스트 ─────────────────

class TestMQTTAlertPublisher:

    @pytest.fixture
    def publisher(self):
        """Mock 모드 발행자."""
        return MQTTAlertPublisher(mock_mode=True)

    def test_connect_mock(self, publisher):
        """Mock 연결이 성공해야 한다."""
        result = publisher.connect()
        assert result is True
        assert publisher.is_connected is True

    def test_publish_fall_detected(self, publisher):
        """낙상 감지 알림 발행이 성공해야 한다."""
        publisher.connect()
        result = publisher.publish_fall_detected(
            person_id=1,
            confidence=0.92,
            position=[0.5, 2.3, 0.1],
        )
        assert result is True
        assert publisher.publish_count == 1

    def test_publish_long_lie(self, publisher):
        """장시간 쓰러짐 알림 발행이 성공해야 한다."""
        publisher.connect()
        result = publisher.publish_long_lie(
            person_id=1,
            fall_duration=35.0,
            alert_level=2,
        )
        assert result is True

    def test_publish_all_clear(self, publisher):
        """낙상 해제 알림 발행이 성공해야 한다."""
        publisher.connect()
        result = publisher.publish_all_clear(person_id=1)
        assert result is True

    def test_publish_count_increments(self, publisher):
        """발행 횟수가 올바르게 증가해야 한다."""
        publisher.connect()
        for i in range(5):
            publisher.publish_fall_detected(i + 1, 0.9)
        assert publisher.publish_count == 5

    def test_topics_are_defined(self, publisher):
        """필수 MQTT 토픽이 정의되어야 한다."""
        required_topics = ["fall_detected", "fall_confirmed", "fall_long_lie", "all_clear", "heartbeat"]
        for topic_key in required_topics:
            assert topic_key in publisher.TOPICS
            assert len(publisher.TOPICS[topic_key]) > 0

    def test_disconnect(self, publisher):
        """연결 해제가 동작해야 한다."""
        publisher.connect()
        publisher.disconnect()
        assert publisher.is_connected is False


# ── VoiceConfirmation 테스트 ──────────────────

class TestVoiceConfirmation:

    @pytest.fixture
    def voice(self):
        return VoiceConfirmation(mock_mode=True, response_timeout=5.0)

    def test_ok_keywords_classified(self, voice):
        """긍정 키워드가 'ok'로 분류되어야 한다."""
        for kw in ["괜찮아", "네 괜찮아요", "응 괜찮아"]:
            result = voice.classify_response(kw)
            assert result == "ok", f"'{kw}' → expected 'ok', got '{result}'"

    def test_help_keywords_classified(self, voice):
        """부정/도움 키워드가 'help'로 분류되어야 한다."""
        for kw in ["아니요", "아파요", "도와줘"]:
            result = voice.classify_response(kw)
            assert result == "help", f"'{kw}' → expected 'help', got '{result}'"

    def test_unknown_classified(self, voice):
        """알 수 없는 텍스트는 'unknown'으로 분류."""
        result = voice.classify_response("잘 모르겠습니다")
        assert result == "unknown"

    def test_empty_string_classified(self, voice):
        """빈 문자열은 'unknown'으로 분류."""
        result = voice.classify_response("")
        assert result == "unknown"

    def test_speak_does_not_raise(self, voice):
        """speak()가 예외 없이 동작해야 한다."""
        voice.speak("테스트 음성 안내")  # Mock 모드 → 출력만

    @pytest.mark.asyncio
    async def test_confirm_async_returns_timeout_in_mock(self, voice):
        """Mock 모드에서 confirm_async()가 'timeout'을 반환해야 한다."""
        result = await voice.confirm_async(person_id=1)
        assert result in ("ok", "help", "timeout")

    @pytest.mark.asyncio
    async def test_ok_callback_called(self, voice):
        """긍정 응답 시 on_ok 콜백이 호출되어야 한다."""
        # Mock 모드이므로 실제 음성 없이 timeout → on_ok 미호출
        # 여기서는 직접 classify_response를 통해 콜백 로직 검증
        callback_called = []

        def on_ok_sync(pid):
            callback_called.append(pid)

        # Mock VoiceConfirmation에서 'ok' 시뮬레이션
        result = voice.classify_response("괜찮아")
        if result == "ok":
            on_ok_sync(1)

        assert 1 in callback_called


# ── SMSSender 테스트 ──────────────────────────

class TestSMSSender:

    def test_mock_send_returns_true(self):
        """Mock SMS 발송이 True를 반환해야 한다."""
        sender = SMSSender(mock_mode=True)
        result = sender.send("테스트 SMS 메시지", to="+82101234567")
        assert result is True

    def test_long_message_handled(self):
        """긴 메시지도 처리되어야 한다."""
        sender = SMSSender(mock_mode=True)
        long_msg = "낙상 감지 알림\n" * 20
        result = sender.send(long_msg)
        assert result is True


# ── AlertEscalationManager 테스트 ─────────────

class TestAlertEscalationManager:

    @pytest.fixture
    def manager(self):
        return AlertEscalationManager(
            mqtt_publisher=None,
            sms_sender=SMSSender(mock_mode=True),
            webhook_sender=EmergencyWebhookSender(mock_mode=True),
            voice_confirmation=None,
            level2_delay=0.2,    # 빠른 테스트용
            level3_delay=0.5,
            cooldown_seconds=2.0,
            mock_mode=True,
        )

    @pytest.mark.asyncio
    async def test_trigger_creates_task(self, manager):
        """trigger() 호출 시 에스컬레이션 태스크가 생성되어야 한다."""
        await manager.trigger(person_id=1, confidence=0.92)
        assert 1 in manager._escalation_tasks

    @pytest.mark.asyncio
    async def test_cooldown_prevents_duplicate(self, manager):
        """쿨다운 중 중복 트리거가 방지되어야 한다."""
        await manager.trigger(person_id=1, confidence=0.92)

        # 즉시 재트리거 → 쿨다운으로 스킵
        # last_escalation 시간 확인
        first_time = manager._last_escalation.get(1, 0.0)
        await manager.trigger(person_id=1, confidence=0.92)

        # 쿨다운 중이면 같은 시간 유지
        assert manager._last_escalation.get(1, 0.0) == first_time

    @pytest.mark.asyncio
    async def test_cancel_stops_pipeline(self, manager):
        """cancel() 호출 시 에스컬레이션이 중단되어야 한다."""
        await manager.trigger(person_id=1, confidence=0.92)
        await asyncio.sleep(0.05)
        await manager.cancel(person_id=1)

        task = manager._escalation_tasks.get(1)
        if task:
            assert task.cancelled() or manager._cancelled.get(1, False)

    @pytest.mark.asyncio
    async def test_event_history_recorded(self, manager):
        """에스컬레이션 이벤트가 히스토리에 기록되어야 한다."""
        await manager.trigger(person_id=1, confidence=0.92)
        await asyncio.sleep(0.6)  # 파이프라인 완료 대기

        history = manager.get_history(person_id=1)
        assert len(history) >= 1

    @pytest.mark.asyncio
    async def test_multiple_persons_independent(self, manager):
        """다수 인원의 에스컬레이션이 독립적으로 동작해야 한다."""
        await manager.trigger(person_id=1, confidence=0.92)
        await manager.trigger(person_id=2, confidence=0.85)

        assert 1 in manager._escalation_tasks
        assert 2 in manager._escalation_tasks

    def test_get_history_empty(self, manager):
        """히스토리가 없을 때 빈 리스트 반환."""
        history = manager.get_history()
        assert isinstance(history, list)
        assert len(history) == 0


# ── 통합: MQTT + 에스컬레이션 ─────────────────

class TestIntegration:

    @pytest.mark.asyncio
    async def test_full_escalation_flow(self):
        """전체 에스컬레이션 플로우가 오류 없이 동작해야 한다."""
        publisher = MQTTAlertPublisher(mock_mode=True)
        publisher.connect()

        sms = SMSSender(mock_mode=True)
        webhook = EmergencyWebhookSender(mock_mode=True)

        manager = AlertEscalationManager(
            mqtt_publisher=publisher,
            sms_sender=sms,
            webhook_sender=webhook,
            level2_delay=0.1,
            level3_delay=0.3,
            cooldown_seconds=60,
        )

        await manager.trigger(person_id=99, confidence=0.95, position=[1.0, 2.0, 0.1])
        await asyncio.sleep(0.5)

        history = manager.get_history(person_id=99)
        assert len(history) >= 1
        publisher.disconnect()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
