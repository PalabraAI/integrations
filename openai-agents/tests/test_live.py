"""
Live tests against the Palabra API (``PALABRA_API_KEY`` required; skipped otherwise).

No OpenAI key is needed: speech for the STT tests is produced by Palabra TTS,
so the suite is self-contained. ``PALABRA_TEST_WAV`` (16-bit mono WAV, any
rate) and ``PALABRA_TEST_LANGUAGE`` swap in your own recording.
"""

from __future__ import annotations

import asyncio
import os
import wave
from pathlib import Path

import numpy as np
import pytest
from agents.voice import AudioInput, StreamedAudioInput, STTModelSettings, TTSModelSettings

from palabra_openai_agents import PalabraSTTModel, PalabraTTSModel

pytestmark = pytest.mark.live

PHRASE = 'The quick brown fox jumps over the lazy dog near the river bank.'
KEYWORDS = ('quick', 'fox', 'lazy', 'dog', 'river')


@pytest.fixture(scope='module')
def spoken_phrase() -> tuple[np.ndarray, int]:
    """PHRASE as pcm16 @ 24 kHz — synthesized once, reused by the STT tests."""

    async def synth():
        model = PalabraTTSModel('en', voice_id='default_low')
        chunks = [c async for c in model.run(PHRASE, TTSModelSettings())]
        return np.frombuffer(b''.join(chunks), dtype=np.int16)

    pcm = asyncio.run(synth())
    assert len(pcm) > 24000, 'TTS produced less than a second of audio'
    assert np.abs(pcm.astype(np.int32)).mean() > 100, 'TTS audio is silent'
    return pcm, 24000


@pytest.fixture(scope='module')
def user_wav() -> tuple[np.ndarray, int, str] | None:
    path = os.environ.get('PALABRA_TEST_WAV')
    if not path:
        return None
    with wave.open(str(Path(path))) as w:
        assert w.getnchannels() == 1, 'PALABRA_TEST_WAV must be mono'
        assert w.getsampwidth() == 2, 'PALABRA_TEST_WAV must be 16-bit'
        return (
            np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16),
            w.getframerate(),
            os.environ.get('PALABRA_TEST_LANGUAGE', 'en'),
        )


async def test_tts_produces_speech(spoken_phrase):
    pcm, rate = spoken_phrase
    assert 2.0 < len(pcm) / rate < 15.0


async def test_stt_batch_native_24k(spoken_phrase):
    pcm, rate = spoken_phrase
    text = await PalabraSTTModel('en').transcribe(
        AudioInput(buffer=pcm, frame_rate=rate), STTModelSettings(), False, False
    )
    print(f'STT batch @24k: {text!r}')
    assert sum(k in text.lower() for k in KEYWORDS) >= 3, text


async def test_stt_batch_autodetect_float32_8k(spoken_phrase):
    pcm, _ = spoken_phrase
    pcm8k = pcm[::3].astype(np.float32) / 32768.0  # crude decimation to 8 kHz float32
    text = await PalabraSTTModel(None).transcribe(
        AudioInput(buffer=pcm8k, frame_rate=8000), STTModelSettings(), False, False
    )
    print(f'STT batch @8k autodetect: {text!r}')
    assert sum(k in text.lower() for k in KEYWORDS) >= 2, text


async def test_stt_streamed_session(spoken_phrase):
    pcm, rate = spoken_phrase
    model = PalabraSTTModel('en', input_frame_rate=rate)
    streamed = StreamedAudioInput()
    session = await model.create_session(streamed, STTModelSettings(), False, False)
    turns: list[str] = []

    async def collect():
        async for turn in session.transcribe_turns():
            print(f'turn: {turn!r}')
            turns.append(turn)

    collector = asyncio.create_task(collect())
    step = int(rate * 0.32)
    for i in range(0, len(pcm), step):  # a live mic: real-time chunks
        await streamed.add_audio(pcm[i : i + step])
        await asyncio.sleep(0.32)
    await streamed.add_audio(None)  # end of speech -> finalize
    await asyncio.sleep(3)
    await session.close()
    await asyncio.wait_for(collector, 5)
    joined = ' '.join(turns).lower()
    assert sum(k in joined for k in KEYWORDS) >= 3, turns


async def test_stt_user_recording(user_wav):
    if user_wav is None:
        pytest.skip('set PALABRA_TEST_WAV to transcribe your own recording')
    pcm, rate, language = user_wav
    text = await PalabraSTTModel(language).transcribe(
        AudioInput(buffer=pcm, frame_rate=rate), STTModelSettings(), False, False
    )
    print(f'STT user wav ({language}, {rate} Hz): {text!r}')
    assert len(text) > 10
