"""Offline tests of the Wyoming server with a scripted fake Palabra client."""

from __future__ import annotations

import asyncio
import socket
from functools import partial

import numpy as np
import pytest
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info
from wyoming.server import AsyncServer
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeStopped, SynthesizeVoice

from wyoming_palabra.handler import HandlerConfig, PalabraEventHandler, _to_pcm_s16le_mono
from wyoming_palabra.info import LANGUAGES, build_info, parse_voice_name

from .conftest import FakePalabra, transcript


def free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture
async def server(request):
    fake = getattr(request, 'param', None) or FakePalabra()
    port = free_port()
    srv = AsyncServer.from_uri(f'tcp://127.0.0.1:{port}')
    config = HandlerConfig(info=build_info(), default_language='en', finalize_timeout_s=2, tts_timeout_s=2)
    task = asyncio.create_task(srv.run(partial(PalabraEventHandler, config, fake)))
    await asyncio.sleep(0.05)
    yield fake, '127.0.0.1', port
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


async def test_describe(server):
    _, host, port = server
    async with AsyncTcpClient(host, port) as client:
        await client.write_event(Describe().event())
        info = Info.from_event(await client.read_event())
    assert info.asr[0].models[0].languages == LANGUAGES
    names = [v.name for v in info.tts[0].voices]
    assert len(names) == 2 * len(LANGUAGES)
    assert {'en-default_low', 'de-default_high'} <= set(names)
    assert next(v for v in info.tts[0].voices if v.name == 'ru-default_low').languages == ['ru']
    assert info.tts[0].supports_synthesize_streaming is True


@pytest.mark.parametrize(
    'server',
    [
        FakePalabra(
            stt_script=[transcript('Turn on', eos=False), transcript('Turn on the lights.')], fin_text='Please.<fin>'
        )
    ],
    indirect=True,
)
async def test_transcribe_streams_native_rate_and_finalizes(server):
    fake, host, port = server
    async with AsyncTcpClient(host, port) as client:
        await client.write_event(Transcribe(language='de-DE').event())
        await client.write_event(AudioStart(rate=44100, width=2, channels=2).event())
        stereo = np.column_stack([np.full(4410, 300, np.int16), np.full(4410, 100, np.int16)]).ravel().tobytes()
        await client.write_event(AudioChunk(rate=44100, width=2, channels=2, audio=stereo).event())
        await client.write_event(AudioStop().event())
        transcript_ev = Transcript.from_event(await client.read_event())
    assert transcript_ev.text == 'Turn on the lights. Please.'
    assert transcript_ev.language == 'de'
    session = fake.stt_sessions[0]
    assert session.params == {'language': 'de', 'sample_rate': 44100}
    assert session.sent == [np.full(4410, 200, np.int16).tobytes()]  # downmixed to mono, not resampled
    assert session.finalized == 1
    assert session.closed


async def test_synthesize_streams_audio_events(server):
    fake, host, port = server
    async with AsyncTcpClient(host, port) as client:
        await client.write_event(
            Synthesize(text='Hello there. Second sentence.', voice=SynthesizeVoice(name='fr-default_high')).event()
        )
        events = [await client.read_event() for _ in range(4)]
    assert AudioStart.is_type(events[0].type)
    chunks = [AudioChunk.from_event(e) for e in events[1:3]]
    assert [c.audio for c in chunks] == [b'Hello there.', b'Second sentence.']
    assert all((c.rate, c.width, c.channels) == (24000, 2, 1) for c in chunks)
    assert AudioStop.is_type(events[3].type)
    assert fake.tts_sessions[0].init['language'] == 'fr'
    assert fake.tts_sessions[0].init['voice_id'] == 'default_high'


async def test_synthesize_streaming_sends_completed_sentences_early(server):
    fake, host, port = server
    async with AsyncTcpClient(host, port) as client:
        await client.write_event(SynthesizeStart(voice=SynthesizeVoice(name='default_low')).event())
        assert AudioStart.is_type((await client.read_event()).type)
        await client.write_event(SynthesizeChunk(text='The lights are ').event())
        await client.write_event(SynthesizeChunk(text='now on. Anything').event())
        await client.write_event(SynthesizeChunk(text=' else?').event())
        await client.write_event(SynthesizeStop().event())
        events = []
        while True:
            ev = await client.read_event()
            events.append(ev)
            if SynthesizeStopped.is_type(ev.type):
                break
    audio = [AudioChunk.from_event(e).audio for e in events if AudioChunk.is_type(e.type)]
    assert audio == [b'The lights are now on.', b'Anything else?']
    assert AudioStop.is_type(events[-2].type)
    session = fake.tts_sessions[0]
    assert [t for t, _ in session.sent] == ['The lights are now on.', 'Anything else?']


def test_to_pcm_s16le_mono_widths():
    assert (
        _to_pcm_s16le_mono(np.array([1, 2], np.int16).tobytes(), width=2, channels=1)
        == np.array([1, 2], np.int16).tobytes()
    )
    assert np.frombuffer(
        _to_pcm_s16le_mono(np.array([1 << 16], np.int32).tobytes(), width=4, channels=1), np.int16
    ).tolist() == [1]
    assert np.frombuffer(_to_pcm_s16le_mono(bytes([128, 255]), width=1, channels=1), np.int16).tolist() == [0, 127 << 8]
    with pytest.raises(ValueError, match='width'):
        _to_pcm_s16le_mono(b'\x00' * 3, width=3, channels=1)


@pytest.mark.parametrize(
    ('name', 'expected'),
    [
        ('en-default_low', ('en', 'default_low')),
        ('default_low', (None, 'default_low')),
        ('xx-voice', (None, 'xx-voice')),
    ],
)
def test_parse_voice_name(name, expected):
    assert parse_voice_name(name) == expected
