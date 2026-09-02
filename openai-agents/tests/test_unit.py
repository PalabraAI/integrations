"""Offline tests of the adapters through the real Agents SDK interfaces."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from agents.voice import AudioInput, StreamedAudioInput, STTModelSettings, TTSModelSettings

from palabra_openai_agents import PalabraSTTModel, PalabraTTSModel, PalabraVoiceModelProvider
from palabra_openai_agents._audio import to_pcm_s16le
from palabra_openai_agents._palabra import Pacer, clamp_speed, short_lang, split_text, strip_fin

from .conftest import FakePalabra, transcript


def test_to_pcm_s16le_int16_passthrough():
    buf = np.array([1, -2, 32767], dtype=np.int16)
    assert to_pcm_s16le(buf) == buf.tobytes()


def test_to_pcm_s16le_float32_scaled_and_clipped():
    out = np.frombuffer(to_pcm_s16le(np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)), dtype=np.int16)
    assert out.tolist() == [0, 32767, -32767, 32767]


def test_to_pcm_s16le_stereo_downmix():
    out = np.frombuffer(to_pcm_s16le(np.array([[100, 300], [0, 0]], dtype=np.int16)), dtype=np.int16)
    assert out.tolist() == [200, 0]


def test_to_pcm_s16le_rejects_other_dtypes():
    with pytest.raises(ValueError, match='int16 or float32'):
        to_pcm_s16le(np.zeros(4, dtype=np.uint8))


@pytest.mark.parametrize(
    ('code', 'expected'), [('en-US', 'en'), ('pt_BR', 'pt'), ('DE', 'de'), (None, None), ('', None)]
)
def test_short_lang(code, expected):
    assert short_lang(code) == expected


def test_strip_fin():
    assert strip_fin('Hello world.<fin>') == 'Hello world.'
    assert strip_fin('<fin>') == ''


def test_split_text_sentences_and_limit():
    assert split_text('  Hello   there.  How are you doing today, friend? Fine. ') == [
        'Hello there.',
        'How are you doing today, friend?',
        'Fine.',
    ]
    assert split_text('Ask Dr. Who about it. Then leave.') == ['Ask Dr. Who about it.', 'Then leave.']
    long = 'word ' * 300
    pieces = split_text(long, limit=100)
    assert all(len(p) <= 100 for p in pieces)
    assert ' '.join(pieces) == long.strip()
    assert split_text('x' * 2500, limit=1024) == ['x' * 1024, 'x' * 1024, 'x' * 452]
    assert split_text('   ') == []


def test_clamp_speed():
    assert clamp_speed(None) is None
    assert clamp_speed(4.0) == 2.0
    assert clamp_speed(0.0) == 0.1
    assert clamp_speed(1.2) == 1.2


async def test_pacer_lets_realtime_through_and_throttles_bursts():
    pacer = Pacer(16000, lead_s=0.1)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    await pacer.wait(3200)  # 100 ms of audio: within the lead, no sleep
    assert loop.time() - t0 < 0.05
    await pacer.wait(16000)  # +500 ms of audio pushed instantly -> must sleep ~0.4 s
    assert loop.time() - t0 >= 0.35


async def test_transcribe_passes_native_rate_and_uses_finalize():
    fake = FakePalabra(stt_script=[transcript('Hello', eos=False), transcript('Hello world.')], fin_text='Bye.<fin>')
    model = PalabraSTTModel('en-GB', client=fake)
    buffer = np.zeros(48000, dtype=np.int16)
    text = await model.transcribe(AudioInput(buffer=buffer, frame_rate=48000), STTModelSettings(), False, False)
    session = fake.stt_sessions[0]
    assert text == 'Hello world. Bye.'
    assert session.params == {'language': 'en', 'sample_rate': 48000}
    assert session.finalized == 1
    assert session.sent == [buffer.tobytes()]  # no resampling, no silence padding
    assert session.closed


async def test_transcribe_settings_language_wins_and_empty_buffer_short_circuits():
    fake = FakePalabra()
    model = PalabraSTTModel('en', client=fake)
    assert (
        await model.transcribe(AudioInput(buffer=np.zeros(0, dtype=np.int16)), STTModelSettings(), False, False) == ''
    )
    assert fake.stt_sessions == []
    settings = STTModelSettings(language='de')
    await model.transcribe(AudioInput(buffer=np.zeros(10, dtype=np.int16)), settings, False, False)
    assert fake.stt_sessions[0].params['language'] == 'de'


async def test_streamed_session_yields_finals_as_turns_and_finalizes_on_end():
    fake = FakePalabra(stt_script=[transcript('partial', eos=False), transcript('First turn.'), transcript('   ')])
    model = PalabraSTTModel(None, client=fake, input_frame_rate=24000)
    streamed = StreamedAudioInput()
    session = await model.create_session(streamed, STTModelSettings(), False, False)
    fake_session = fake.stt_sessions[0]
    assert fake_session.params == {'language': None, 'sample_rate': 24000}

    turns: list[str] = []

    async def collect():
        async for turn in session.transcribe_turns():
            turns.append(turn)

    collector = asyncio.create_task(collect())
    await streamed.add_audio(np.zeros(2400, dtype=np.int16))
    await streamed.add_audio(np.zeros(2400, dtype=np.float32))
    await streamed.add_audio(None)  # end of input -> finalize
    await asyncio.sleep(0.3)
    await session.close()
    await asyncio.wait_for(collector, 2)  # transcribe_turns() returns after close()
    assert turns == ['First turn.']
    assert fake_session.finalized == 1
    assert [len(c) for c in fake_session.sent] == [4800, 4800]
    assert fake_session.closed


async def test_streamed_session_turn_gap_merges_finals():
    fake = FakePalabra(stt_script=[transcript('One.'), transcript('Two.')])
    model = PalabraSTTModel('en', client=fake, turn_gap_s=0.3)
    session = await model.create_session(StreamedAudioInput(), STTModelSettings(), False, False)
    turns = []

    async def collect():
        async for turn in session.transcribe_turns():
            turns.append(turn)

    collector = asyncio.create_task(collect())
    await asyncio.sleep(0.6)
    await session.close()
    await asyncio.wait_for(collector, 2)
    assert turns == ['One. Two.']


async def test_tts_streams_pcm_in_order_and_honours_speed():
    fake = FakePalabra()
    model = PalabraTTSModel('en-US', voice_id='default_high', client=fake)
    settings = TTSModelSettings(speed=3.0)
    chunks = [c async for c in model.run('First sentence here. Second sentence follows it.', settings)]
    session = fake.tts_sessions[0]
    assert b''.join(chunks) == b'First sentence here.Second sentence follows it.'
    expected_init = {'language': 'en', 'voice_id': 'default_high', 'speed': 2.0, 'format': 'pcm', 'sample_rate': 24000}
    assert session.init == expected_init
    assert [eos for _, eos in session.sent] == [True, True]


async def test_tts_empty_text_opens_no_session():
    fake = FakePalabra()
    assert [c async for c in PalabraTTSModel('en', client=fake).run('  ', TTSModelSettings())] == []
    assert fake.tts_sessions == []


def test_tts_requires_language():
    with pytest.raises(ValueError, match='language'):
        PalabraTTSModel('', client=FakePalabra())


def test_provider_hands_out_palabra_models():
    provider = PalabraVoiceModelProvider(stt_language='de', tts_language='de', client=FakePalabra())
    assert provider.get_stt_model('whatever').model_name == 'palabra-stt'
    assert provider.get_tts_model(None).model_name == 'palabra-tts'
