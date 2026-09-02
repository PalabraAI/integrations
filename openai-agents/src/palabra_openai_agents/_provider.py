"""``VoiceModelProvider`` so ``VoicePipelineConfig`` can hand out Palabra models by name."""

from __future__ import annotations

from agents.voice import STTModel, TTSModel, VoiceModelProvider
from palabra_ai import Palabra

from ._stt import PalabraSTTModel
from ._tts import DEFAULT_VOICE_ID, PalabraTTSModel


class PalabraVoiceModelProvider(VoiceModelProvider):
    """
    Use Palabra for both STT and TTS via ``VoicePipelineConfig(model_provider=...)``.

    Model names passed by the pipeline are accepted but ignored: there is one
    Palabra STT and one Palabra TTS model, configured here.
    """

    def __init__(
        self,
        *,
        stt_language: str | None = None,
        tts_language: str = 'en',
        voice_id: str = DEFAULT_VOICE_ID,
        api_key: str | None = None,
        client: Palabra | None = None,
    ):
        self._client = client or Palabra(api_key=api_key)
        self._stt_language = stt_language
        self._tts_language = tts_language
        self._voice_id = voice_id

    def get_stt_model(self, model_name: str | None) -> STTModel:
        return PalabraSTTModel(self._stt_language, client=self._client)

    def get_tts_model(self, model_name: str | None) -> TTSModel:
        return PalabraTTSModel(self._tts_language, voice_id=self._voice_id, client=self._client)
