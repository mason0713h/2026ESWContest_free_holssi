"""
alert/voice_confirm.py
======================
음성 확인 모듈.

낙상 감지 후 스피커로 한국어 음성 안내를 출력하고,
마이크로 사용자의 음성 응답을 인식하여 에스컬레이션 여부를 결정한다.

TTS: pyttsx3 (오프라인) 또는 espeak (Linux 시스템 TTS)
STT: openai-whisper tiny 모델 (오프라인, 한국어)

응답 키워드:
    긍정 (자력 확인): "괜찮", "괜찮아", "괜찮아요", "네", "예", "좋아요"
    부정 (도움 요청): "아니요", "아니", "아파", "못 움직여", "도와줘"
    미응답 → 에스컬레이션 트리거
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import threading
import time
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# espeak-ng 가용성 확인 (한국어 TTS 1순위 — pyttsx3는 Linux에서 음성팩이
# 등록돼 있어도 한국어 음성을 찾지 못한 채 예외 없이 영어로 발화하는
# 경우가 많아 신뢰할 수 없다. espeak-ng -v ko 직접 호출이 더 안정적이다.)
ESPEAK_NG_PATH = shutil.which("espeak-ng")
ESPEAK_AVAILABLE = ESPEAK_NG_PATH is not None
if not ESPEAK_AVAILABLE:
    logger.warning("espeak-ng 미설치 (sudo apt-get install espeak-ng). pyttsx3로 폴백")

# TTS 가용성 확인 (espeak-ng 없을 때의 2차 폴백)
try:
    import pyttsx3  # type: ignore
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logger.warning("pyttsx3 미설치. Mock TTS만 사용 가능")

# STT 가용성 확인
try:
    import whisper  # type: ignore
    STT_AVAILABLE = True
except ImportError:
    STT_AVAILABLE = False
    logger.warning("openai-whisper 미설치. Mock STT 사용")

# 마이크 입력 가용성 확인
try:
    import sounddevice as sd  # type: ignore
    import numpy as np
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False
    np = None
    logger.warning("sounddevice 미설치. Mock 마이크 사용")


# ──────────────────────────────────────────────
# 음성 안내 메시지
# ──────────────────────────────────────────────

VOICE_MESSAGES = {
    "initial": (
        "낙상이 감지되었습니다. "
        "괜찮으시면 30초 안에 괜찮아라고 말씀해 주세요. "
        "응답이 없으면 보호자에게 연락드립니다."
    ),
    "reminder": (
        "다시 한번 여쭙겠습니다. 괜찮으세요? "
        "괜찮으시면 괜찮아라고 말씀해 주세요."
    ),
    "escalating": (
        "응답이 없어 보호자에게 연락 중입니다. "
        "잠시만 기다려 주세요."
    ),
    "all_clear": "네, 알겠습니다. 안심되네요. 필요하시면 언제든지 부르세요.",
    "help_coming": "도움을 요청하겠습니다. 잠시만 기다려 주세요.",
}

OK_KEYWORDS = {"괜찮", "괜찮아", "괜찮아요", "네", "예", "응", "좋아", "좋아요"}
HELP_KEYWORDS = {"아니", "아니요", "아파", "못", "도와줘", "살려줘", "아파요"}


class VoiceConfirmation:
    """
    낙상 후 음성 확인 기능.

    TTS로 안내 메시지를 출력하고 STT로 응답을 인식한다.
    timeout 내에 긍정 응답이 없으면 에스컬레이션 콜백을 호출한다.

    Args:
        response_timeout: 음성 응답 대기 시간 (초, 기본 30)
        sample_rate: 마이크 샘플레이트 (Hz)
        record_duration: 한 번 녹음할 시간 (초)
        whisper_model_size: Whisper 모델 크기 ('tiny', 'base', 'small')
        mock_mode: True면 실제 오디오 없이 시뮬레이션
    """

    def __init__(
        self,
        response_timeout: float = 30.0,
        sample_rate: int = 16000,
        record_duration: float = 3.0,
        whisper_model_size: str = "tiny",
        mock_mode: bool = False,
    ) -> None:
        self.response_timeout = response_timeout
        self.sample_rate = sample_rate
        self.record_duration = record_duration
        self.mock_mode = mock_mode or (not MIC_AVAILABLE and not STT_AVAILABLE)

        self._tts_engine = None
        self._whisper_model = None
        self._tts_lock = threading.Lock()

        if not self.mock_mode:
            self._init_tts()
            self._init_stt(whisper_model_size)

        logger.info(
            "VoiceConfirmation 초기화: timeout=%.0fs, mock=%s",
            response_timeout, self.mock_mode,
        )

    def _init_tts(self) -> None:
        """TTS 엔진 초기화. espeak-ng(한국어 직접 지원)를 1순위로 사용한다."""
        if ESPEAK_AVAILABLE:
            logger.info("espeak-ng 한국어 TTS 사용 (%s)", ESPEAK_NG_PATH)
            return

        if TTS_AVAILABLE:
            try:
                self._tts_engine = pyttsx3.init()
                voices = self._tts_engine.getProperty("voices")
                korean_voice = next(
                    (v for v in voices if "ko" in v.id.lower() or "korean" in v.name.lower()),
                    None,
                )
                if korean_voice is not None:
                    self._tts_engine.setProperty("voice", korean_voice.id)
                else:
                    logger.warning(
                        "pyttsx3에서 한국어 음성을 찾지 못함. 영어/기본 음성으로 발화될 수 있음 "
                        "(espeak-ng 설치 권장: sudo apt-get install espeak-ng)"
                    )
                self._tts_engine.setProperty("rate", 150)
                self._tts_engine.setProperty("volume", 0.9)
                logger.info("pyttsx3 TTS 초기화 완료")
            except Exception as e:
                logger.warning("pyttsx3 초기화 실패: %s", e)
                self._tts_engine = None
        else:
            logger.warning("TTS 엔진 없음 (espeak-ng, pyttsx3 모두 미설치)")

    def _init_stt(self, model_size: str) -> None:
        """Whisper STT 모델 초기화."""
        if STT_AVAILABLE:
            try:
                logger.info("Whisper '%s' 모델 로드 중...", model_size)
                self._whisper_model = whisper.load_model(model_size)
                logger.info("Whisper 모델 로드 완료")
            except Exception as e:
                logger.warning("Whisper 모델 로드 실패: %s", e)
                self._whisper_model = None

    def speak(self, text: str) -> None:
        """
        TTS로 텍스트 음성 출력.

        Args:
            text: 출력할 텍스트
        """
        logger.info("[TTS] %s", text)

        if self.mock_mode:
            print(f"  [TTS] {text}")
            return

        with self._tts_lock:
            if ESPEAK_AVAILABLE:
                try:
                    import subprocess
                    subprocess.run(
                        [ESPEAK_NG_PATH, "-v", "ko", "-s", "140", text],
                        timeout=10,
                        capture_output=True,
                        check=False,
                    )
                    return
                except Exception as e:
                    logger.warning("espeak-ng 발화 오류: %s", e)

            # pyttsx3 폴백 (한국어 음성이 없으면 영어로 발화될 수 있음)
            if self._tts_engine is not None:
                try:
                    self._tts_engine.say(text)
                    self._tts_engine.runAndWait()
                except Exception as e:
                    logger.warning("pyttsx3 발화 오류: %s", e)

    def record_audio(self) -> Optional[object]:
        """
        마이크에서 오디오 녹음.

        Returns:
            numpy 배열 (float32, mono) 또는 None
        """
        if self.mock_mode or not MIC_AVAILABLE:
            return None

        try:
            import numpy as np
            logger.debug("%.1f초 녹음 시작", self.record_duration)
            audio = sd.rec(
                int(self.sample_rate * self.record_duration),
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            return audio.flatten()
        except Exception as e:
            logger.error("녹음 오류: %s", e)
            return None

    def transcribe(self, audio) -> str:
        """
        Whisper로 음성 → 텍스트 변환.

        Args:
            audio: numpy float32 배열

        Returns:
            인식된 텍스트 (소문자)
        """
        if self.mock_mode or self._whisper_model is None or audio is None:
            return ""

        try:
            import numpy as np
            result = self._whisper_model.transcribe(
                audio.astype(np.float32),
                language="ko",
                task="transcribe",
                fp16=False,
            )
            text = result.get("text", "").strip()
            logger.info("[STT] 인식: '%s'", text)
            return text
        except Exception as e:
            logger.error("STT 오류: %s", e)
            return ""

    def classify_response(self, text: str) -> str:
        """
        인식된 텍스트를 응답 유형으로 분류.

        Args:
            text: STT 인식 텍스트

        Returns:
            'ok' | 'help' | 'unknown'
        """
        text_lower = text.lower().replace(" ", "")
        for kw in OK_KEYWORDS:
            if kw in text_lower:
                return "ok"
        for kw in HELP_KEYWORDS:
            if kw in text_lower:
                return "help"
        return "unknown"

    async def confirm_async(
        self,
        person_id: int,
        on_ok: Optional[callable] = None,
        on_escalate: Optional[callable] = None,
    ) -> str:
        """
        비동기 음성 확인 루프.

        안내 메시지를 출력하고 timeout 내에 응답을 기다린다.
        응답 유형에 따라 on_ok 또는 on_escalate 콜백을 호출한다.

        Args:
            person_id: 대상 인원 ID
            on_ok: 긍정 응답 시 콜백 (person_id)
            on_escalate: 미응답/부정 시 콜백 (person_id, reason)

        Returns:
            최종 응답 유형: 'ok' | 'help' | 'timeout'
        """
        loop = asyncio.get_event_loop()
        response_result = {"type": "timeout"}

        def _confirm_blocking():
            """블로킹 음성 확인 (executor에서 실행)."""
            # 초기 안내
            self.speak(VOICE_MESSAGES["initial"])
            start_time = time.time()
            attempt = 0

            while time.time() - start_time < self.response_timeout:
                elapsed = time.time() - start_time
                remaining = self.response_timeout - elapsed

                # 중간 리마인드 (15초 경과 시)
                if attempt == 0 and elapsed >= 15:
                    self.speak(VOICE_MESSAGES["reminder"])
                    attempt += 1

                if self.mock_mode:
                    # Mock: 응답 없음 시뮬레이션 (timeout 테스트)
                    time.sleep(min(5.0, remaining))
                    continue

                # 녹음 및 인식
                audio = self.record_audio()
                if audio is not None:
                    text = self.transcribe(audio)
                    if text:
                        response_type = self.classify_response(text)
                        if response_type == "ok":
                            self.speak(VOICE_MESSAGES["all_clear"])
                            response_result["type"] = "ok"
                            logger.info("인원 #%d 자력 확인 응답", person_id)
                            return
                        elif response_type == "help":
                            self.speak(VOICE_MESSAGES["help_coming"])
                            response_result["type"] = "help"
                            logger.warning("인원 #%d 도움 요청", person_id)
                            return

            # 타임아웃
            self.speak(VOICE_MESSAGES["escalating"])
            response_result["type"] = "timeout"
            logger.warning("인원 #%d 음성 응답 없음 (%.0f초)", person_id, self.response_timeout)

        # executor에서 블로킹 작업 실행
        await loop.run_in_executor(None, _confirm_blocking)

        result_type = response_result["type"]

        # 콜백 호출
        if result_type == "ok" and on_ok:
            if asyncio.iscoroutinefunction(on_ok):
                await on_ok(person_id)
            else:
                on_ok(person_id)
        elif result_type in ("help", "timeout") and on_escalate:
            if asyncio.iscoroutinefunction(on_escalate):
                await on_escalate(person_id, result_type)
            else:
                on_escalate(person_id, result_type)

        return result_type

    def confirm_sync(self, person_id: int, timeout: Optional[float] = None) -> str:
        """
        동기 음성 확인 (블로킹).

        Args:
            person_id: 대상 인원 ID
            timeout: 응답 대기 시간 (None이면 self.response_timeout 사용)

        Returns:
            'ok' | 'help' | 'timeout'
        """
        if timeout is not None:
            original_timeout = self.response_timeout
            self.response_timeout = timeout

        self.speak(VOICE_MESSAGES["initial"])
        start = time.time()

        while time.time() - start < self.response_timeout:
            if self.mock_mode:
                time.sleep(self.response_timeout)  # 즉시 timeout
                break

            audio = self.record_audio()
            if audio is not None:
                text = self.transcribe(audio)
                response_type = self.classify_response(text)
                if response_type in ("ok", "help"):
                    if timeout is not None:
                        self.response_timeout = original_timeout
                    return response_type

        if timeout is not None:
            self.response_timeout = original_timeout
        return "timeout"


# ──────────────────────────────────────────────
# 단독 실행 테스트
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)
    print("=== VoiceConfirmation 테스트 (Mock 모드) ===\n")

    voice = VoiceConfirmation(
        response_timeout=5.0,   # 테스트용: 5초
        mock_mode=True,
    )

    # 응답 분류 테스트
    test_texts = [
        ("네 괜찮아요", "ok"),
        ("아니요 아파요", "help"),
        ("뭐라고요", "unknown"),
        ("괜찮아", "ok"),
    ]
    print("[응답 분류 테스트]")
    for text, expected in test_texts:
        result = voice.classify_response(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{text}' → {result} (예상: {expected})")

    # 비동기 확인 테스트
    print("\n[비동기 확인 테스트 (Mock - Timeout 시나리오)]")

    async def test_async():
        escalated = []

        async def on_escalate(person_id, reason):
            escalated.append((person_id, reason))
            print(f"  에스컬레이션 트리거: person_id={person_id}, reason={reason}")

        result = await voice.confirm_async(
            person_id=1,
            on_escalate=on_escalate,
        )
        print(f"  최종 결과: {result}")
        return result

    import time
    t0 = time.perf_counter()
    final = asyncio.run(test_async())
    elapsed = time.perf_counter() - t0
    print(f"  소요 시간: {elapsed:.1f}초")
    print(f"\n테스트 완료!")
