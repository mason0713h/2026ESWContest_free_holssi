"""
alert/mqtt_publisher.py
=======================
MQTT 알림 발행 모듈.

낙상 감지 이벤트를 MQTT 브로커로 발행한다.
보호자 앱, SMS 게이트웨이, 119 연계 시스템이 구독하여 알림을 수신한다.

토픽 구조:
    fall_guardian/fall/detected    - 낙상 감지 즉시 알림
    fall_guardian/fall/confirmed   - 음성 확인 후 확정 알림
    fall_guardian/fall/long_lie    - 장시간 쓰러짐 알림
    fall_guardian/status/clear     - 낙상 해제 알림
    fall_guardian/status/heartbeat - 시스템 상태 주기 발행
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# paho-mqtt 가용성 확인
try:
    import paho.mqtt.client as mqtt  # type: ignore
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    logger.warning("paho-mqtt 미설치. Mock MQTT 모드로 동작합니다.")


# ──────────────────────────────────────────────
# 메시지 데이터 클래스
# ──────────────────────────────────────────────

@dataclass
class FallAlertMessage:
    """
    낙상 알림 MQTT 메시지 페이로드.

    Attributes:
        event_type: 이벤트 유형 (fall_detected, confirmed, long_lie, clear)
        person_id: 대상 인원 ID
        timestamp: 이벤트 발생 시각 (Unix)
        confidence: 낙상 감지 신뢰도
        position: 인원 위치 [x, y, z]
        fall_duration: 낙상 후 경과 시간 (초)
        alert_level: 알림 레벨 (1~4)
        device_id: 기기 ID
        message_ko: 한국어 메시지
    """
    event_type: str
    person_id: int
    timestamp: float
    confidence: float = 0.0
    position: Optional[list] = None
    fall_duration: float = 0.0
    alert_level: int = 1
    device_id: str = "fall_guardian_001"
    message_ko: str = ""

    def to_json(self) -> str:
        """JSON 직렬화."""
        d = asdict(self)
        d["timestamp_iso"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp)
        )
        return json.dumps(d, ensure_ascii=False)


# ──────────────────────────────────────────────
# MQTT 발행자
# ──────────────────────────────────────────────

class MQTTAlertPublisher:
    """
    MQTT 브로커에 낙상 알림을 발행하는 클라이언트.

    Jetson Orin Nano에서 실행되어 로컬 Mosquitto 브로커에 연결한다.

    Args:
        broker_host: MQTT 브로커 호스트
        broker_port: MQTT 브로커 포트
        client_id: MQTT 클라이언트 ID
        username: MQTT 인증 사용자명 (선택)
        password: MQTT 인증 비밀번호 (선택)
        use_tls: TLS 암호화 사용 여부
        qos: QoS 레벨 (0, 1, 2)
        keepalive: keepalive 인터벌 (초)
        device_id: 기기 식별자
        mock_mode: True면 실제 연결 없이 로그만 출력
    """

    TOPICS = {
        "fall_detected": "fall_guardian/fall/detected",
        "fall_confirmed": "fall_guardian/fall/confirmed",
        "fall_long_lie": "fall_guardian/fall/long_lie",
        "all_clear": "fall_guardian/status/clear",
        "heartbeat": "fall_guardian/status/heartbeat",
    }

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        client_id: str = "fall_guardian_001",
        username: str = "",
        password: str = "",
        use_tls: bool = False,
        qos: int = 1,
        keepalive: int = 60,
        device_id: str = "fall_guardian_001",
        mock_mode: bool = False,
    ) -> None:
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client_id = client_id
        self.qos = qos
        self.keepalive = keepalive
        self.device_id = device_id
        self.mock_mode = mock_mode or not MQTT_AVAILABLE

        self._client: Optional[Any] = None
        self._connected = False
        self._publish_count = 0
        self._last_publish_time: Dict[str, float] = {}

        if not self.mock_mode:
            self._setup_client(username, password, use_tls)

        logger.info(
            "MQTTAlertPublisher 초기화: %s:%d (mock=%s)",
            broker_host, broker_port, self.mock_mode,
        )

    def _setup_client(self, username: str, password: str, use_tls: bool) -> None:
        """paho-mqtt 클라이언트 설정."""
        self._client = mqtt.Client(client_id=self.client_id, clean_session=True)

        if username:
            self._client.username_pw_set(username, password)

        if use_tls:
            self._client.tls_set()

        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_publish = self._on_publish

    def _on_connect(self, client, userdata, flags, rc) -> None:
        """연결 콜백."""
        if rc == 0:
            self._connected = True
            logger.info("MQTT 브로커 연결 성공: %s:%d", self.broker_host, self.broker_port)
        else:
            logger.error("MQTT 연결 실패: rc=%d", rc)

    def _on_disconnect(self, client, userdata, rc) -> None:
        """연결 해제 콜백."""
        self._connected = False
        logger.warning("MQTT 연결 해제: rc=%d", rc)
        if rc != 0:
            logger.info("자동 재연결 시도 중...")

    def _on_publish(self, client, userdata, mid) -> None:
        """발행 완료 콜백."""
        logger.debug("MQTT 메시지 발행 완료: mid=%d", mid)

    def connect(self) -> bool:
        """
        MQTT 브로커에 연결.

        Returns:
            연결 성공 여부
        """
        if self.mock_mode:
            logger.info("[Mock] MQTT 연결 시뮬레이션")
            self._connected = True
            return True

        try:
            self._client.connect(
                self.broker_host,
                self.broker_port,
                keepalive=self.keepalive,
            )
            self._client.loop_start()
            # 연결 대기 (최대 5초)
            timeout = 5.0
            start = time.time()
            while not self._connected and time.time() - start < timeout:
                time.sleep(0.1)
            return self._connected
        except Exception as e:
            logger.error("MQTT 연결 오류: %s", e)
            return False

    def disconnect(self) -> None:
        """MQTT 연결 해제."""
        if not self.mock_mode and self._client:
            self._client.loop_stop()
            self._client.disconnect()
        self._connected = False

    def publish(self, topic: str, payload: str, retain: bool = False) -> bool:
        """
        MQTT 메시지 발행.

        Args:
            topic: MQTT 토픽
            payload: 발행할 JSON 페이로드
            retain: retain 플래그

        Returns:
            발행 성공 여부
        """
        if self.mock_mode:
            logger.info("[Mock MQTT] 토픽=%s | %s", topic, payload[:100])
            self._publish_count += 1
            return True

        if not self._connected:
            logger.error("MQTT 미연결 상태에서 발행 시도")
            return False

        try:
            result = self._client.publish(topic, payload, qos=self.qos, retain=retain)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self._publish_count += 1
                return True
            else:
                logger.error("MQTT 발행 실패: rc=%d", result.rc)
                return False
        except Exception as e:
            logger.error("MQTT 발행 오류: %s", e)
            return False

    # ── 이벤트별 발행 메서드 ────────────────────

    def publish_fall_detected(
        self,
        person_id: int,
        confidence: float,
        position: Optional[list] = None,
    ) -> bool:
        """
        낙상 감지 즉시 알림 발행.

        Args:
            person_id: 낙상 감지 인원 ID
            confidence: 낙상 신뢰도
            position: 인원 위치 [x, y, z]

        Returns:
            발행 성공 여부
        """
        msg = FallAlertMessage(
            event_type="fall_detected",
            person_id=person_id,
            timestamp=time.time(),
            confidence=confidence,
            position=position or [0.0, 0.0, 0.0],
            alert_level=1,
            device_id=self.device_id,
            message_ko=f"낙상이 감지되었습니다. (인원 #{person_id}, 신뢰도: {confidence:.1%})",
        )
        logger.warning("낙상 감지 알림: person_id=%d, confidence=%.3f", person_id, confidence)
        return self.publish(self.TOPICS["fall_detected"], msg.to_json())

    def publish_fall_confirmed(
        self,
        person_id: int,
        confidence: float,
        position: Optional[list] = None,
    ) -> bool:
        """음성 확인 후 낙상 확정 알림 발행."""
        msg = FallAlertMessage(
            event_type="fall_confirmed",
            person_id=person_id,
            timestamp=time.time(),
            confidence=confidence,
            position=position or [0.0, 0.0, 0.0],
            alert_level=2,
            device_id=self.device_id,
            message_ko=f"낙상이 확인되었습니다. 도움이 필요합니다. (인원 #{person_id})",
        )
        return self.publish(self.TOPICS["fall_confirmed"], msg.to_json())

    def publish_long_lie(
        self,
        person_id: int,
        fall_duration: float,
        alert_level: int,
        position: Optional[list] = None,
    ) -> bool:
        """장시간 쓰러짐 알림 발행."""
        msg = FallAlertMessage(
            event_type="long_lie",
            person_id=person_id,
            timestamp=time.time(),
            fall_duration=fall_duration,
            alert_level=alert_level,
            position=position or [0.0, 0.0, 0.0],
            device_id=self.device_id,
            message_ko=(
                f"장시간 쓰러짐 감지: {fall_duration:.0f}초 경과 "
                f"(인원 #{person_id}, 레벨 {alert_level})"
            ),
        )
        logger.warning(
            "장시간 쓰러짐 알림: person_id=%d, duration=%.0fs, level=%d",
            person_id, fall_duration, alert_level,
        )
        return self.publish(self.TOPICS["fall_long_lie"], msg.to_json())

    def publish_all_clear(self, person_id: int) -> bool:
        """낙상 해제 알림 발행."""
        msg = FallAlertMessage(
            event_type="all_clear",
            person_id=person_id,
            timestamp=time.time(),
            alert_level=0,
            device_id=self.device_id,
            message_ko=f"낙상 해제: 인원 #{person_id}이(가) 회복하였습니다.",
        )
        return self.publish(self.TOPICS["all_clear"], msg.to_json())

    async def publish_heartbeat_loop(self, interval: float = 30.0) -> None:
        """
        시스템 상태 heartbeat를 주기적으로 발행 (비동기).

        Args:
            interval: 발행 간격 (초)
        """
        while True:
            heartbeat = json.dumps({
                "device_id": self.device_id,
                "timestamp": time.time(),
                "status": "running",
                "total_published": self._publish_count,
            })
            self.publish(self.TOPICS["heartbeat"], heartbeat)
            await asyncio.sleep(interval)

    @property
    def is_connected(self) -> bool:
        """연결 상태 확인."""
        return self._connected

    @property
    def publish_count(self) -> int:
        """총 발행 횟수."""
        return self._publish_count


# ──────────────────────────────────────────────
# 단독 실행 테스트
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)
    print("=== MQTTAlertPublisher 테스트 (Mock 모드) ===\n")

    publisher = MQTTAlertPublisher(
        broker_host="localhost",
        broker_port=1883,
        mock_mode=True,
    )

    assert publisher.connect(), "연결 실패"
    print("연결 성공")

    # 낙상 감지 알림
    publisher.publish_fall_detected(
        person_id=1,
        confidence=0.94,
        position=[0.5, 2.3, 0.15],
    )

    # 음성 확인 후 확정
    import time
    time.sleep(0.5)
    publisher.publish_fall_confirmed(person_id=1, confidence=0.94)

    # 장시간 쓰러짐
    time.sleep(0.5)
    publisher.publish_long_lie(person_id=1, fall_duration=32.5, alert_level=2)

    # Heartbeat
    async def test_heartbeat():
        task = asyncio.create_task(publisher.publish_heartbeat_loop(interval=0.5))
        await asyncio.sleep(1.5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(test_heartbeat())

    print(f"\n총 발행 횟수: {publisher.publish_count}")
    publisher.disconnect()
    print("테스트 완료!")
