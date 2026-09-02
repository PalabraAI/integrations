"""Shared fixtures: a scripted fake of the palabra-ai client for offline unit tests."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field

import pytest
from palabra_ai import SttTranscript, TtsChunk


def pytest_collection_modifyitems(config, items):
    if os.environ.get('PALABRA_API_KEY'):
        return
    skip = pytest.mark.skip(reason='live test: set PALABRA_API_KEY to run')
    for item in items:
        if 'live' in item.keywords:
            item.add_marker(skip)


def transcript(text: str, *, eos: bool = True) -> SttTranscript:
    return SttTranscript(
        type='transcription', text=text, language='en', transcription_id='t1', is_eos=eos, is_translation=False
    )


def tts_chunk(audio: bytes, gen: str, *, last: bool = False) -> TtsChunk:
    return TtsChunk(type='audio_chunk', audio=audio, generation_id=gen, last_chunk=last)


class FakeSttSession:
    """Records sent audio; replays scripted events; answers `finalize` with a <fin> final."""

    def __init__(self, script: list, fin_text: str = '<fin>'):
        self.script = list(script)
        self.fin_text = fin_text
        self.sent: list[bytes] = []
        self.finalized = 0
        self.closed = False
        self.params: dict = {}
        self._events: asyncio.Queue = asyncio.Queue()
        self._ws = self  # `finalize()` falls back to session._ws.send(...)

    async def __aenter__(self):
        for ev in self.script:
            self._events.put_nowait(ev)
        return self

    async def __aexit__(self, *exc):
        await self.close()

    async def send(self, raw: str) -> None:  # the ws stand-in
        assert '"finalize"' in raw
        self.finalized += 1
        self._events.put_nowait(transcript(self.fin_text, eos=self.fin_text != '<fin>'))

    async def send_audio(self, chunk: bytes) -> None:
        self.sent.append(chunk)

    async def send_pcm(self, pcm: bytes, *, realtime: bool = True) -> None:
        self.sent.append(pcm)

    async def receive(self, timeout=None):
        if self.closed:
            return None
        if timeout is None:
            return await self._events.get()
        return await asyncio.wait_for(self._events.get(), timeout)

    async def close(self) -> None:
        self.closed = True
        self._events.put_nowait(None)


class FakeTtsSession:
    """Emits one audio chunk (the text, encoded) + last_chunk per sent text."""

    def __init__(self):
        self.sent: list[tuple[str, bool]] = []
        self.init: dict = {}
        self._events: asyncio.Queue = asyncio.Queue()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def send_text(self, text: str, *, eos: bool = False, generation_id: str | None = None, **_):
        assert text
        assert len(text) <= 1024
        self.sent.append((text, eos))
        gen = generation_id or 'gen'
        self._events.put_nowait(tts_chunk(text.encode(), gen))
        self._events.put_nowait(tts_chunk(b'', gen, last=True))

    async def receive(self, timeout=None):
        return await asyncio.wait_for(self._events.get(), timeout or 5)

    async def close(self) -> None:
        return None


@dataclass
class FakePalabra:
    stt_script: list = field(default_factory=list)
    stt_sessions: list = field(default_factory=list)
    tts_sessions: list = field(default_factory=list)
    fin_text: str = '<fin>'

    def stt(self, language=None, *, sample_rate=None, **kw):
        session = FakeSttSession(self.stt_script, self.fin_text)
        session.params = {'language': language, 'sample_rate': sample_rate, **kw}
        self.stt_sessions.append(session)
        return session

    def tts(self, language, **kw):
        session = FakeTtsSession()
        session.init = {'language': language, **kw}
        self.tts_sessions.append(session)
        return session

    def for_key(self, _key):
        return self
