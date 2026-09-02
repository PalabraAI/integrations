"""
Live test of the Wyoming server against the Palabra API (skipped without PALABRA_API_KEY).

A real Wyoming client (the same class Home Assistant uses) synthesizes a phrase,
then streams the audio back as a ``transcribe`` request.
"""

from __future__ import annotations

import asyncio
import os
import socket
from functools import partial

import pytest
from palabra_ai import Palabra
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info
from wyoming.server import AsyncServer
from wyoming.tts import Synthesize, SynthesizeVoice

from wyoming_palabra.handler import HandlerConfig, PalabraEventHandler
from wyoming_palabra.info import build_info

pytestmark = pytest.mark.live

PHRASE = 'Turn on the kitchen lights and set the thermostat to twenty one degrees.'
KEYWORDS = ('kitchen', 'lights', 'thermostat', 'twenty', 'degrees')


@pytest.fixture
async def server():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    srv = AsyncServer.from_uri(f'tcp://127.0.0.1:{port}')
    config = HandlerConfig(info=build_info(), default_language='en')
    task = asyncio.create_task(
        srv.run(partial(PalabraEventHandler, config, Palabra(api_key=os.environ['PALABRA_API_KEY'])))
    )
    await asyncio.sleep(0.05)
    yield '127.0.0.1', port
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


async def test_describe_synthesize_transcribe(server):
    host, port = server
    async with AsyncTcpClient(host, port) as client:
        await client.write_event(Describe().event())
        info = Info.from_event(await client.read_event())
        assert info.tts[0].voices

        await client.write_event(Synthesize(text=PHRASE, voice=SynthesizeVoice(name='en-default_low')).event())
        start = AudioStart.from_event(await client.read_event())
        pcm = bytearray()
        while True:
            ev = await asyncio.wait_for(client.read_event(), 30)
            if AudioStop.is_type(ev.type):
                break
            pcm.extend(AudioChunk.from_event(ev).audio)
        seconds = len(pcm) / 2 / start.rate
        print(f'synthesize: {seconds:.1f}s @ {start.rate} Hz')
        assert seconds > 2

        # play it back as a voice command, at the TTS rate (24 kHz) — the server resamples
        await client.write_event(Transcribe(language='en').event())
        await client.write_event(AudioStart(rate=start.rate, width=2, channels=1).event())
        step = int(start.rate * 0.32) * 2
        for i in range(0, len(pcm), step):
            await client.write_event(
                AudioChunk(rate=start.rate, width=2, channels=1, audio=bytes(pcm[i : i + step])).event()
            )
            await asyncio.sleep(0.32)
        await client.write_event(AudioStop().event())
        text = Transcript.from_event(await asyncio.wait_for(client.read_event(), 30)).text
    print(f'transcript: {text!r}')
    assert sum(k in text.lower() for k in KEYWORDS) >= 3, text
