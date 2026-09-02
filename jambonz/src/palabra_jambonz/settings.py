"""Runtime configuration, read from environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    palabra_api_key: str | None = None
    #: The API key configured for the custom vendor in the jambonz portal; if
    #: set, ``Authorization: Bearer <key>`` is required on STT and TTS.
    jambonz_api_key: str | None = None
    #: Fallback voice when jambonz sends none.
    voice_id: str = 'default_low'
    #: Sample rate of the WAV returned to jambonz (Palabra TTS: 8000-48000).
    tts_sample_rate: int = 24000
    #: How long to wait for the final transcript after ``stop``.
    finalize_timeout_s: float = 10.0
    log_level: str = 'INFO'

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        return cls(
            palabra_api_key=env.get('PALABRA_API_KEY') or None,
            jambonz_api_key=env.get('JAMBONZ_API_KEY') or None,
            voice_id=env.get('PALABRA_VOICE_ID', 'default_low'),
            tts_sample_rate=int(env.get('PALABRA_TTS_SAMPLE_RATE', '24000')),
            finalize_timeout_s=float(env.get('PALABRA_FINALIZE_TIMEOUT', '10')),
            log_level=env.get('LOG_LEVEL', 'INFO').upper(),
        )
