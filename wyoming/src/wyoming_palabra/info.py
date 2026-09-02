"""Wyoming ``describe`` -> ``info`` payload: what this server offers Home Assistant."""

from __future__ import annotations

from wyoming.info import AsrModel, AsrProgram, Attribution, Info, TtsProgram, TtsVoice

from . import __version__

#: Languages supported by Palabra Realtime STT and TTS.
LANGUAGES = ['ar', 'de', 'en', 'es', 'fr', 'hi', 'it', 'ja', 'ko', 'nl', 'pt', 'ru', 'zh']
#: Built-in voices; cloned voices from the Palabra Voices API can be added via ``--voice``.
DEFAULT_VOICES = ['default_low', 'default_high']

ATTRIBUTION = Attribution(name='Palabra.ai', url='https://palabra.ai')


def voice_name(language: str, voice_id: str) -> str:
    """
    Advertised voice name: ``en-default_low``.

    Wyoming's ``synthesize`` carries either a voice *name* or a language, never
    both, so the language is folded into the name — Home Assistant then offers
    only the voices matching the assistant's language and the server knows
    which language to synthesize in.
    """
    return f'{language}-{voice_id}'


def parse_voice_name(name: str) -> tuple[str | None, str]:
    """``en-default_low`` -> ``('en', 'default_low')``; a bare voice id -> ``(None, voice_id)``."""
    language, sep, voice_id = name.partition('-')
    if sep and language in LANGUAGES and voice_id:
        return language, voice_id
    return None, name


def build_info(voices: list[str] | None = None) -> Info:
    voices = voices or DEFAULT_VOICES
    return Info(
        asr=[
            AsrProgram(
                name='palabra',
                description='Palabra Realtime Speech-to-Text',
                attribution=ATTRIBUTION,
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name='palabra-stt',
                        description='Palabra realtime ASR (server-side VAD and resampling)',
                        attribution=ATTRIBUTION,
                        installed=True,
                        version=__version__,
                        languages=LANGUAGES,
                    )
                ],
                supports_transcript_streaming=False,
            )
        ],
        tts=[
            TtsProgram(
                name='palabra',
                description='Palabra Realtime Text-to-Speech',
                attribution=ATTRIBUTION,
                installed=True,
                version=__version__,
                voices=[
                    TtsVoice(
                        name=voice_name(language, voice),
                        description=f'Palabra voice {voice} ({language})',
                        attribution=ATTRIBUTION,
                        installed=True,
                        version=__version__,
                        languages=[language],
                    )
                    for voice in voices
                    for language in LANGUAGES
                ],
                supports_synthesize_streaming=True,
            )
        ],
    )
