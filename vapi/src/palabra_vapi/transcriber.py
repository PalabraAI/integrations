"""
Vapi Custom Transcriber (WebSocket) backed by Palabra Realtime STT.

Protocol (https://docs.vapi.ai/customization/custom-transcriber): Vapi opens a
WebSocket, sends ``{"type": "start", "encoding": "linear16", "container": "raw",
"sampleRate": 16000, "channels": 2}`` and then binary PCM frames — interleaved
stereo, channel 0 = customer, channel 1 = assistant. We answer with
``{"type": "transcriber-response", "transcription", "channel", "transcriptType"}``.

Each channel is fed to its own Palabra STT session at the call's native sample
rate (the ASR resamples server-side). One API Key supports one live session, so
the assistant channel is transcribed only when a second key is configured.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from palabra_ai import Palabra, PalabraError, SttSession, SttTranscript

from ._palabra import FIN_MARKER, Pacer, finalize, short_lang, strip_fin
from .settings import Settings

log = logging.getLogger(__name__)

CHANNEL_NAMES = ('customer', 'assistant')
SUPPORTED_ENCODINGS = {'linear16'}


@dataclass
class _Channel:
    name: str
    session: SttSession
    pacer: Pacer
    forwarder: asyncio.Task
    flushed: asyncio.Event = field(default_factory=asyncio.Event)


class TranscriberConnection:
    """One Vapi call: parses control messages, fans audio out per channel, forwards transcripts."""

    def __init__(self, ws: WebSocket, settings: Settings, client_factory: Callable[[str | None], Palabra]):
        self._ws = ws
        self._settings = settings
        self._client_factory = client_factory
        self._sample_rate = 16000
        self._channels = 2
        self._streams: dict[str, _Channel] = {}
        self._send_lock = asyncio.Lock()

    async def run(self) -> None:
        try:
            while True:
                message = await self._ws.receive()
                if message['type'] == 'websocket.disconnect':
                    break
                if (text := message.get('text')) is not None:
                    await self._on_control(text)
                elif (data := message.get('bytes')) is not None:
                    await self._on_audio(data)
        except WebSocketDisconnect:
            pass
        except PalabraError as e:
            log.error('Palabra STT failed: %s', e)
            await self._close(code=1011, reason=str(e)[:120])
        finally:
            await self._shutdown()

    async def _on_control(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            log.warning('ignoring non-JSON text frame')
            return
        kind = msg.get('type')
        if kind == 'start':
            await self._start(msg)
        else:
            log.debug('ignoring control message %r', kind)

    async def _start(self, msg: dict) -> None:
        if self._streams:
            log.warning('duplicate start message ignored')
            return
        encoding = str(msg.get('encoding', 'linear16')).lower()
        if encoding not in SUPPORTED_ENCODINGS:
            log.error('unsupported encoding %r', encoding)
            await self._close(code=1003, reason=f'unsupported encoding {encoding}')
            raise WebSocketDisconnect
        self._sample_rate = int(msg.get('sampleRate', 16000))
        self._channels = int(msg.get('channels', 2))
        log.info('call started: %d Hz, %d channel(s)', self._sample_rate, self._channels)

        keys = {'customer': self._settings.palabra_api_key}
        if self._channels >= 2:
            if self._settings.assistant_api_key:
                keys['assistant'] = self._settings.assistant_api_key
            else:
                log.info('assistant channel not transcribed: set PALABRA_API_KEY_ASSISTANT to enable')
        language = short_lang(self._settings.stt_language)
        for name, key in keys.items():
            session = self._client_factory(key).stt(language=language, sample_rate=self._sample_rate)
            await session.__aenter__()
            channel = _Channel(name=name, session=session, pacer=Pacer(self._sample_rate), forwarder=None)  # type: ignore[arg-type]
            channel.forwarder = asyncio.create_task(self._forward(channel), name=f'vapi-forward-{name}')
            self._streams[name] = channel

    async def _on_audio(self, data: bytes) -> None:
        if not self._streams:
            return  # audio before start: nothing to feed yet
        frame = np.frombuffer(data[: len(data) - len(data) % (2 * self._channels)], dtype=np.int16)
        if self._channels >= 2:
            per_channel = {name: frame[i :: self._channels] for i, name in enumerate(CHANNEL_NAMES)}
        else:
            per_channel = {'customer': frame}
        for name, pcm in per_channel.items():
            stream = self._streams.get(name)
            if stream is None or not len(pcm):
                continue
            chunk = pcm.tobytes()
            await stream.pacer.wait(len(chunk))
            await stream.session.send_audio(chunk)

    async def _forward(self, channel: _Channel) -> None:
        """Palabra transcripts -> Vapi ``transcriber-response`` messages."""
        try:
            while True:
                ev = await channel.session.receive()
                if ev is None:
                    log.info('Palabra STT session for %s ended', channel.name)
                    return
                if not isinstance(ev, SttTranscript) or ev.is_translation:
                    continue
                answers_finalize = FIN_MARKER in ev.text
                text = strip_fin(ev.text)
                if text and (ev.is_eos or self._settings.send_partials):
                    await self._send_json(
                        {
                            'type': 'transcriber-response',
                            'transcription': text,
                            'channel': channel.name,
                            'transcriptType': 'final' if ev.is_eos else 'partial',
                        }
                    )
                if answers_finalize:
                    channel.flushed.set()
        finally:
            channel.flushed.set()

    async def _send_json(self, payload: dict) -> None:
        async with self._send_lock:
            with contextlib.suppress(RuntimeError, WebSocketDisconnect):  # peer already gone
                await self._ws.send_text(json.dumps(payload))

    async def _shutdown(self) -> None:
        """Flush the recognizers so the last words of the call are not lost, then close."""
        if not self._streams:
            return
        try:
            await asyncio.wait_for(self._flush_all(), timeout=self._settings.finalize_timeout_s)
        except asyncio.TimeoutError:
            log.warning('recognizer did not flush within %.0fs', self._settings.finalize_timeout_s)
        for stream in self._streams.values():
            stream.forwarder.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await stream.forwarder
            await stream.session.close()
        self._streams.clear()

    async def _flush_all(self) -> None:
        async def flush(stream: _Channel) -> None:
            with contextlib.suppress(Exception):  # best effort: the call is over anyway
                await finalize(stream.session)
                await stream.flushed.wait()  # the forwarder relays the finals answering finalize

        await asyncio.gather(*(flush(s) for s in self._streams.values()))

    async def _close(self, *, code: int, reason: str) -> None:
        with contextlib.suppress(Exception):
            await self._ws.close(code=code, reason=reason)


async def handle_transcriber(
    ws: WebSocket, settings: Settings, client_factory: Callable[[str | None], Palabra]
) -> None:
    if settings.vapi_secret and ws.headers.get('x-vapi-secret') != settings.vapi_secret:
        log.warning('transcriber: rejected connection with bad x-vapi-secret')
        await ws.close(code=1008, reason='invalid x-vapi-secret')
        return
    await ws.accept()
    await TranscriberConnection(ws, settings, client_factory).run()
