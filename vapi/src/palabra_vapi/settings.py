"""Runtime configuration, read from environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

_TRUE = {'1', 'true', 'yes', 'on'}


@dataclass(frozen=True)
class Settings:
    """All knobs of the bridge. See ``from_env`` for the variable names."""

    palabra_api_key: str | None = None
    #: A second API Key for the assistant channel: one key supports one live STT session.
    assistant_api_key: str | None = None
    #: If set, ``x-vapi-secret`` must match on every request.
    vapi_secret: str | None = None
    #: STT source language; ``None`` = server-side auto-detection.
    stt_language: str | None = None
    #: Also forward interim transcripts as ``transcriptType: "partial"``.
    send_partials: bool = False
    tts_language: str = 'en'
    voice_id: str = 'default_low'
    #: How long to wait for the recognizer to flush after the call audio stops.
    finalize_timeout_s: float = 10.0
    log_level: str = 'INFO'

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        return cls(
            palabra_api_key=env.get('PALABRA_API_KEY') or None,
            assistant_api_key=env.get('PALABRA_API_KEY_ASSISTANT') or None,
            vapi_secret=env.get('VAPI_SECRET') or None,
            stt_language=env.get('PALABRA_STT_LANGUAGE') or env.get('PALABRA_LANGUAGE') or None,
            send_partials=env.get('VAPI_SEND_PARTIALS', '').lower() in _TRUE,
            tts_language=env.get('PALABRA_TTS_LANGUAGE', 'en'),
            voice_id=env.get('PALABRA_VOICE_ID', 'default_low'),
            finalize_timeout_s=float(env.get('PALABRA_FINALIZE_TIMEOUT', '10')),
            log_level=env.get('LOG_LEVEL', 'INFO').upper(),
        )
