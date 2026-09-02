"""
Small helpers on top of the ``palabra-ai`` client used by this integration.

Everything here is protocol glue for the Palabra Realtime STT/TTS APIs
(https://platform.palabra.ai/docs): the ``finalize`` command and its ``<fin>``
marker, real-time pacing of audio, and sentence splitting under the TTS text
limit. The audio itself is never resampled — the ASR endpoint accepts any PCM
sample rate via the ``sample_rate`` parameter and resamples server-side.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Iterable

from palabra_ai import Palabra, ServerError, SessionError, SttSession, SttTranscript, TaskError, TtsChunk

log = logging.getLogger('palabra_openai_agents')

#: Appended by the recognizer to the final transcript that answers a ``finalize`` command.
FIN_MARKER = '<fin>'
#: Server limit for one TTS ``text`` message.
MAX_TTS_TEXT = 1024
#: Palabra TTS accepts speeds in this range (the Agents SDK allows 0.25-4.0).
TTS_SPEED_RANGE = (0.1, 2.0)

_SENTENCE_END = re.compile(r'(?<=[.!?…。！？])\s+')
_MIN_FRAGMENT = 12  # a "sentence" shorter than this ("Dr.", "No.") is glued to the next one


def short_lang(code: str | None) -> str | None:
    """``'en-US'`` -> ``'en'``; ``None`` (or empty) keeps server-side auto-detection."""
    if not code:
        return None
    return re.split(r'[-_]', code, maxsplit=1)[0].lower()


def strip_fin(text: str) -> str:
    return text.replace(FIN_MARKER, '').strip()


async def finalize(session: SttSession) -> None:
    """
    Ask the recognizer to finalize everything received so far.

    Works in both finalization modes; the final transcript answering the
    command carries :data:`FIN_MARKER` (``"<fin>"`` alone when nothing was
    recognized).
    """
    method = getattr(session, 'finalize', None)
    if method is not None:  # newer palabra-ai releases expose it directly
        await method()
        return
    ws = getattr(session, '_ws', None)  # palabra-ai <= 2.1 has no wrapper yet
    if ws is None:
        raise SessionError('STT session is not connected')
    await ws.send(json.dumps({'message_type': 'finalize'}))


async def collect_finals(session: SttSession, *, timeout: float) -> list[str]:
    """Final transcripts until the one answering ``finalize`` arrives (or ``timeout``)."""
    finals: list[str] = []
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while (remaining := deadline - loop.time()) > 0:
        try:
            ev = await session.receive(timeout=remaining)
        except asyncio.TimeoutError:
            log.warning('STT: no final transcript within %.1fs of finalize', timeout)
            break
        if ev is None:
            break
        if not isinstance(ev, SttTranscript) or ev.is_translation:
            continue
        done = FIN_MARKER in ev.text
        if ev.is_eos and (text := strip_fin(ev.text)):
            finals.append(text)
        if done:
            break
    return finals


class Pacer:
    """
    Throttle a PCM stream to real time.

    The realtime recognizer consumes audio at wall-clock speed; frames pushed
    much faster than that (a file replayed as fast as the network allows) are
    not transcribed. Live sources — a microphone, a phone call — are already
    real-time and are never delayed; ``lead_s`` of slack absorbs their jitter.
    """

    def __init__(self, sample_rate: int, *, channels: int = 1, sample_width: int = 2, lead_s: float = 0.5):
        self._bytes_per_s = sample_rate * channels * sample_width
        self._lead_s = lead_s
        self._started: float | None = None
        self._sent = 0

    async def wait(self, nbytes: int) -> None:
        """Call before sending ``nbytes`` of audio; sleeps if we are ahead of real time."""
        now = time.monotonic()
        if self._started is None:
            self._started = now
        self._sent += nbytes
        ahead = self._started + self._sent / self._bytes_per_s - now
        if ahead > self._lead_s:
            await asyncio.sleep(ahead - self._lead_s)


def split_text(text: str, limit: int = MAX_TTS_TEXT) -> list[str]:
    """
    Split ``text`` into sentences no longer than ``limit`` characters.

    One sentence per TTS message keeps latency low (audio for the first
    sentence starts before the rest is synthesized); over-long sentences are
    split on words, and tiny fragments ("Dr.") are merged with what follows.
    """
    text = ' '.join(text.split())
    if not text:
        return []
    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        if pieces and len(pieces[-1]) < _MIN_FRAGMENT and len(pieces[-1]) + 1 + len(sentence) <= limit:
            pieces[-1] = f'{pieces[-1]} {sentence}'
            continue
        pieces.extend(_split_long(sentence, limit))
    return pieces


def _split_long(sentence: str, limit: int) -> Iterable[str]:
    if len(sentence) <= limit:
        yield sentence
        return
    current = ''
    for word in sentence.split():
        while len(word) > limit:
            if current:
                yield current
                current = ''
            yield word[:limit]
            word = word[limit:]
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= limit:
            current = f'{current} {word}'
        else:
            yield current
            current = word
    if current:
        yield current


def clamp_speed(speed: float | None) -> float | None:
    if speed is None:
        return None
    lo, hi = TTS_SPEED_RANGE
    return min(max(float(speed), lo), hi)


async def synthesize(
    client: Palabra,
    text: str,
    *,
    language: str,
    voice_id: str,
    sample_rate: int,
    speed: float | None = None,
    timeout: float = 60.0,
) -> AsyncIterator[bytes]:
    """
    Stream pcm_s16le mono audio for ``text`` over one TTS session.

    All sentences are queued up-front (the server synthesizes them in order)
    and the audio is yielded generation by generation, so chunks never
    interleave across sentences. Raises ``TaskError`` on a server ``error``
    message and ``SessionError`` if the connection drops mid-synthesis.
    """
    pieces = split_text(text)
    if not pieces:
        return
    async with client.tts(
        language, voice_id=voice_id, speed=clamp_speed(speed), format='pcm', sample_rate=sample_rate
    ) as session:
        pending = [uuid.uuid4().hex[:12] for _ in pieces]
        for gen, piece in zip(pending, pieces, strict=True):
            await session.send_text(piece, eos=True, generation_id=gen)
        buffered: dict[str, bytearray] = {}
        finished: set[str] = set()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while pending:
            current = pending[0]
            if current in finished:
                pending.pop(0)
                if pending and (buf := buffered.pop(pending[0], None)):
                    yield bytes(buf)
                continue
            try:
                ev = await session.receive(timeout=max(deadline - loop.time(), 0.001))
            except asyncio.TimeoutError as e:
                raise SessionError(f'TTS produced no audio within {timeout:.0f}s') from e
            if ev is None:
                raise SessionError('TTS connection closed mid-synthesis')
            if isinstance(ev, ServerError):
                raise TaskError(ev.code, ev.desc)
            if not isinstance(ev, TtsChunk):
                continue
            deadline = loop.time() + timeout  # progress resets the stall timer
            if ev.generation_id == current:
                if ev.audio:
                    yield ev.audio
            elif ev.audio:
                buffered.setdefault(ev.generation_id, bytearray()).extend(ev.audio)
            if ev.last_chunk:
                finished.add(ev.generation_id)
