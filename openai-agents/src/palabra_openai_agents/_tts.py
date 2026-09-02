"""Palabra Realtime TTS as an OpenAI Agents SDK ``TTSModel``."""

from __future__ import annotations

from collections.abc import AsyncIterator

from agents.voice import TTSModel, TTSModelSettings
from palabra_ai import Palabra

from ._palabra import short_lang, synthesize

#: ``VoicePipeline`` expects pcm_s16le mono at this rate (the OpenAI TTS model's format).
OUTPUT_SAMPLE_RATE = 24000
DEFAULT_VOICE_ID = 'default_low'


class PalabraTTSModel(TTSModel):
    """
    Text-to-speech through the Palabra Realtime TTS WebSocket API.

    Emits pcm_s16le mono at 24 kHz — exactly what ``VoicePipeline`` expects.
    ``TTSModelSettings.voice`` (OpenAI voice names) is ignored; choose a
    Palabra voice with ``voice_id`` (``default_low``, ``default_high`` or a
    cloned voice from the Voices API). ``TTSModelSettings.speed`` is honoured
    (clamped to the Palabra range) unless ``speed`` is given here.

    Args:
        language: language of the text to synthesize (``"en"``, ``"de-DE"`` ...).
        voice_id: Palabra voice.
        speed: speaking rate override, 0.1-2.0.
        api_key: Palabra API Key; defaults to ``$PALABRA_API_KEY``.
        client: a pre-configured :class:`palabra_ai.Palabra`.

    """

    def __init__(
        self,
        language: str,
        *,
        voice_id: str = DEFAULT_VOICE_ID,
        speed: float | None = None,
        api_key: str | None = None,
        client: Palabra | None = None,
    ):
        language_code = short_lang(language)
        if not language_code:
            raise ValueError('PalabraTTSModel requires a language, e.g. "en"')
        self._client = client or Palabra(api_key=api_key)
        self._language = language_code
        self._voice_id = voice_id
        self._speed = speed

    @property
    def model_name(self) -> str:
        return 'palabra-tts'

    async def run(self, text: str, settings: TTSModelSettings) -> AsyncIterator[bytes]:
        speed = self._speed if self._speed is not None else settings.speed
        async for chunk in synthesize(
            self._client,
            text,
            language=self._language,
            voice_id=self._voice_id,
            sample_rate=OUTPUT_SAMPLE_RATE,
            speed=speed,
        ):
            yield chunk
