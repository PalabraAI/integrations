"""One Wyoming client connection: ASR (transcribe/audio-*) and TTS (synthesize, synthesize-*)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import uuid
from dataclasses import dataclass

import numpy as np
from palabra_ai import Palabra, PalabraError, ServerError, SttSession, TaskError, TtsChunk, TtsSession
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeStopped, SynthesizeVoice

from ._palabra import Pacer, collect_finals, finalize, short_lang, split_text
from .info import parse_voice_name

log = logging.getLogger(__name__)

TTS_RATE = 24000
TTS_WIDTH = 2
TTS_CHANNELS = 1
_SENTENCE_BOUNDARY = re.compile(r'[.!?…。！？]+(?=\s|$)')


@dataclass(frozen=True)
class HandlerConfig:
    info: Info
    default_language: str = 'en'
    default_voice: str = 'default_low'
    finalize_timeout_s: float = 10.0
    tts_timeout_s: float = 60.0


@dataclass
class _Recognition:
    session: SttSession
    pacer: Pacer
    rate: int
    width: int
    channels: int


class _StreamingSynthesis:
    """synthesize-start .. synthesize-chunk* .. synthesize-stop over one TTS session."""

    def __init__(self, session: TtsSession):
        self.session = session
        self.buffer = ''
        self.pending: list[str] = []  # generation ids in order

    async def feed(self, text: str, *, flush: bool = False) -> None:
        """Queue completed sentences for synthesis; keep the unfinished tail until more text (or ``flush``)."""
        self.buffer += text
        if flush:
            head, self.buffer = self.buffer, ''
        else:
            boundaries = list(_SENTENCE_BOUNDARY.finditer(self.buffer))
            if not boundaries:
                return
            end = boundaries[-1].end()
            head, self.buffer = self.buffer[:end], self.buffer[end:]
        for sentence in split_text(head):
            gen = uuid.uuid4().hex[:12]
            self.pending.append(gen)
            await self.session.send_text(sentence, eos=True, generation_id=gen)


class PalabraEventHandler(AsyncEventHandler):
    def __init__(self, config: HandlerConfig, client: Palabra, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config = config
        self._client = client
        self._language: str | None = None
        self._recognition: _Recognition | None = None
        self._synthesis: _StreamingSynthesis | None = None

    async def handle_event(self, event: Event) -> bool:
        try:
            return await self._dispatch(event)
        except PalabraError as e:
            log.error('Palabra request failed: %s', e)
            await self.write_event(Error(text=str(e), code=type(e).__name__).event())
            await self._abort()
            return True

    async def _dispatch(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self._config.info.event())
            return True
        if Transcribe.is_type(event.type):
            self._language = Transcribe.from_event(event).language
            return True
        if AudioStart.is_type(event.type):
            await self._audio_start(AudioStart.from_event(event))
            return True
        if AudioChunk.is_type(event.type):
            await self._audio_chunk(AudioChunk.from_event(event))
            return True
        if AudioStop.is_type(event.type):
            await self._audio_stop()
            return True
        if Synthesize.is_type(event.type):
            await self._synthesize(Synthesize.from_event(event))
            return True
        if SynthesizeStart.is_type(event.type):
            await self._synthesize_start(SynthesizeStart.from_event(event))
            return True
        if SynthesizeChunk.is_type(event.type):
            if self._synthesis is not None:
                await self._synthesis.feed(SynthesizeChunk.from_event(event).text)
            return True
        if SynthesizeStop.is_type(event.type):
            await self._synthesize_stop()
            return True
        log.debug('ignoring event %s', event.type)
        return True

    # ---- ASR -------------------------------------------------------------

    async def _audio_start(self, start: AudioStart) -> None:
        await self._abort_recognition()
        language = short_lang(self._language)
        log.info(
            'transcribe: %s, %d Hz, %d ch, %d-bit', language or 'auto', start.rate, start.channels, start.width * 8
        )
        session = self._client.stt(language=language, sample_rate=start.rate)
        await session.__aenter__()
        self._recognition = _Recognition(
            session=session, pacer=Pacer(start.rate), rate=start.rate, width=start.width, channels=start.channels
        )

    async def _audio_chunk(self, chunk: AudioChunk) -> None:
        rec = self._recognition
        if rec is None:  # HA always sends audio-start first; tolerate clients that don't
            await self._audio_start(AudioStart(rate=chunk.rate, width=chunk.width, channels=chunk.channels))
            rec = self._recognition
        pcm = _to_pcm_s16le_mono(chunk.audio, width=chunk.width or rec.width, channels=chunk.channels or rec.channels)
        if not pcm:
            return
        await rec.pacer.wait(len(pcm))
        await rec.session.send_audio(pcm)

    async def _audio_stop(self) -> None:
        rec = self._recognition
        if rec is None:
            await self.write_event(Transcript(text='').event())
            return
        self._recognition = None
        try:
            await finalize(rec.session)
            finals = await collect_finals(rec.session, timeout=self._config.finalize_timeout_s)
        finally:
            await rec.session.close()
        text = ' '.join(finals)
        log.info('transcript: %r', text)
        await self.write_event(Transcript(text=text, language=short_lang(self._language)).event())

    # ---- TTS -------------------------------------------------------------

    def _voice_and_language(self, voice: SynthesizeVoice | None) -> tuple[str, str]:
        """Resolve (voice_id, language) from the request, the last ``transcribe`` or the defaults."""
        voice_language, voice_id = parse_voice_name(voice.name) if voice and voice.name else (None, None)
        language = (
            voice_language
            or short_lang(voice.language if voice and voice.language else None)
            or short_lang(self._language)
            or self._config.default_language
        )
        return voice_id or self._config.default_voice, language

    async def _synthesize(self, synth: Synthesize) -> None:
        voice_id, language = self._voice_and_language(synth.voice)
        log.info('synthesize: %d chars (%s, %s)', len(synth.text), language, voice_id)
        async with self._client.tts(language, voice_id=voice_id, format='pcm', sample_rate=TTS_RATE) as session:
            streaming = _StreamingSynthesis(session)
            await self.write_event(AudioStart(rate=TTS_RATE, width=TTS_WIDTH, channels=TTS_CHANNELS).event())
            await streaming.feed(synth.text, flush=True)
            await self._relay_audio(streaming)
        await self.write_event(AudioStop().event())

    async def _synthesize_start(self, start: SynthesizeStart) -> None:
        await self._abort_synthesis()
        voice_id, language = self._voice_and_language(start.voice)
        log.info('synthesize-start (%s, %s)', language, voice_id)
        session = self._client.tts(language, voice_id=voice_id, format='pcm', sample_rate=TTS_RATE)
        await session.__aenter__()
        self._synthesis = _StreamingSynthesis(session)
        await self.write_event(AudioStart(rate=TTS_RATE, width=TTS_WIDTH, channels=TTS_CHANNELS).event())

    async def _synthesize_stop(self) -> None:
        streaming = self._synthesis
        if streaming is None:
            await self.write_event(SynthesizeStopped().event())
            return
        self._synthesis = None
        try:
            await streaming.feed('', flush=True)
            await self._relay_audio(streaming)
        finally:
            await streaming.session.close()
        await self.write_event(AudioStop().event())
        await self.write_event(SynthesizeStopped().event())

    async def _relay_audio(self, streaming: _StreamingSynthesis) -> None:
        """Forward audio for every queued generation, in order, until all are complete."""
        pending = list(streaming.pending)
        streaming.pending.clear()
        buffered: dict[str, bytearray] = {}
        finished: set[str] = set()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._config.tts_timeout_s
        while pending:
            current = pending[0]
            if current in finished:
                pending.pop(0)
                if pending and (buf := buffered.pop(pending[0], None)):
                    await self._write_audio(bytes(buf))
                continue
            try:
                ev = await streaming.session.receive(timeout=max(deadline - loop.time(), 0.001))
            except asyncio.TimeoutError as e:
                raise TaskError('TIMEOUT', f'no audio within {self._config.tts_timeout_s:.0f}s') from e
            if ev is None:
                raise TaskError('CONNECTION_CLOSED', 'TTS connection closed mid-synthesis')
            if isinstance(ev, ServerError):
                raise TaskError(ev.code, ev.desc)
            if not isinstance(ev, TtsChunk):
                continue
            deadline = loop.time() + self._config.tts_timeout_s
            if ev.generation_id == current:
                if ev.audio:
                    await self._write_audio(ev.audio)
            elif ev.audio:
                buffered.setdefault(ev.generation_id, bytearray()).extend(ev.audio)
            if ev.last_chunk:
                finished.add(ev.generation_id)

    async def _write_audio(self, pcm: bytes) -> None:
        await self.write_event(AudioChunk(rate=TTS_RATE, width=TTS_WIDTH, channels=TTS_CHANNELS, audio=pcm).event())

    # ---- lifecycle ---------------------------------------------------------

    async def _abort_recognition(self) -> None:
        if self._recognition is not None:
            rec, self._recognition = self._recognition, None
            with contextlib.suppress(Exception):
                await rec.session.close()

    async def _abort_synthesis(self) -> None:
        if self._synthesis is not None:
            synth, self._synthesis = self._synthesis, None
            with contextlib.suppress(Exception):
                await synth.session.close()

    async def _abort(self) -> None:
        await self._abort_recognition()
        await self._abort_synthesis()

    async def disconnect(self) -> None:
        await self._abort()


def _to_pcm_s16le_mono(audio: bytes, *, width: int, channels: int) -> bytes:
    """Wyoming audio (any width, any channel count) -> pcm_s16le mono."""
    if width == 2:
        samples = np.frombuffer(audio[: len(audio) - len(audio) % (2 * channels)], dtype=np.int16)
    elif width == 4:
        raw = np.frombuffer(audio[: len(audio) - len(audio) % (4 * channels)], dtype=np.int32)
        samples = (raw >> 16).astype(np.int16)
    elif width == 1:
        raw = np.frombuffer(audio[: len(audio) - len(audio) % channels], dtype=np.uint8)
        samples = ((raw.astype(np.int16) - 128) << 8).astype(np.int16)
    else:
        raise ValueError(f'unsupported sample width {width}')
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return samples.tobytes()
