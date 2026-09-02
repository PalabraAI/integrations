"""
jambonz custom STT vendor (WebSocket) backed by Palabra Realtime STT.

Protocol (https://docs.jambonz.org/guides/features/custom-stt-providers):
jambonz connects with ``Authorization: Bearer <apiKey>`` (no subprotocol),
sends ``{"type": "start", "language", "format": "raw", "encoding": "LINEAR16",
"sampleRateHz", "interimResults", "options"}``, then binary linear16 frames,
then ``{"type": "stop"}`` — after which the vendor returns the final transcript
and closes the socket. Responses are ``transcription`` messages
(``is_final``, ``alternatives[{transcript, confidence}]``, ``channel``,
``language``) and ``{"type": "error", "error"}``.

Audio is forwarded at the call's sample rate (8 kHz telephony as a rule);
the ASR resamples server-side.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable

from fastapi import WebSocket, WebSocketDisconnect
from palabra_ai import Palabra, PalabraError, SttSession, SttTranscript

from ._palabra import FIN_MARKER, Pacer, finalize, short_lang, strip_fin
from .settings import Settings

log = logging.getLogger(__name__)

SUPPORTED_ENCODINGS = {'linear16'}
#: Palabra does not report per-utterance confidence; jambonz requires a 0-1 value.
CONFIDENCE = 0.9


class SttConnection:
    """One jambonz recognition stream."""

    def __init__(self, ws: WebSocket, settings: Settings, client_factory: Callable[[str | None], Palabra]):
        self._ws = ws
        self._settings = settings
        self._client_factory = client_factory
        self._session: SttSession | None = None
        self._forwarder: asyncio.Task | None = None
        self._pacer: Pacer | None = None
        self._flushed = asyncio.Event()
        self._sample_rate = 8000
        self._language = 'en-US'
        self._interim = False

    async def run(self) -> None:
        try:
            while True:
                message = await self._ws.receive()
                if message['type'] == 'websocket.disconnect':
                    break
                if (text := message.get('text')) is not None:
                    if await self._on_control(text) is False:
                        break
                elif (data := message.get('bytes')) is not None and self._session is not None:
                    await self._pacer.wait(len(data))
                    await self._session.send_audio(data)
        except WebSocketDisconnect:
            pass
        except PalabraError as e:
            log.error('Palabra STT failed: %s', e)
            await self._send_json({'type': 'error', 'error': str(e)})
        finally:
            await self._teardown()
            with contextlib.suppress(Exception):
                await self._ws.close()

    async def _on_control(self, raw: str) -> bool | None:
        """Returns False when the stream is over (``stop`` handled)."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            log.warning('ignoring non-JSON text frame')
            return None
        kind = msg.get('type')
        if kind == 'start':
            await self._start(msg)
        elif kind == 'stop':
            await self._stop()
            return False
        else:
            log.debug('ignoring control message %r', kind)
        return None

    async def _start(self, msg: dict) -> None:
        if self._session is not None:
            log.warning('duplicate start message ignored')
            return
        encoding = str(msg.get('encoding', 'LINEAR16')).lower()
        if encoding not in SUPPORTED_ENCODINGS:
            await self._send_json({'type': 'error', 'error': f'unsupported encoding {encoding}'})
            raise WebSocketDisconnect
        self._sample_rate = int(msg.get('sampleRateHz', 8000))
        self._language = str(msg.get('language') or 'en-US')
        self._interim = bool(msg.get('interimResults', False))
        if msg.get('options', {}).get('hints'):
            log.debug('hints are not supported by Palabra STT and are ignored')
        log.info('recognition started: %s, %d Hz, interim=%s', self._language, self._sample_rate, self._interim)
        self._pacer = Pacer(self._sample_rate)
        self._session = self._client_factory(self._settings.palabra_api_key).stt(
            language=short_lang(self._language), sample_rate=self._sample_rate
        )
        await self._session.__aenter__()
        self._forwarder = asyncio.create_task(self._forward(self._session), name='jambonz-forward')

    async def _stop(self) -> None:
        """Deliver the final transcript, as the protocol requires, before we close."""
        if self._session is None:
            return
        try:
            await finalize(self._session)
            await asyncio.wait_for(self._flushed.wait(), timeout=self._settings.finalize_timeout_s)
        except asyncio.TimeoutError:
            log.warning('no final transcript within %.0fs of stop', self._settings.finalize_timeout_s)

    async def _forward(self, session: SttSession) -> None:
        try:
            while True:
                ev = await session.receive()
                if ev is None:
                    log.info('Palabra STT session ended')
                    return
                if not isinstance(ev, SttTranscript) or ev.is_translation:
                    continue
                answers_finalize = FIN_MARKER in ev.text
                text = strip_fin(ev.text)
                if text and (ev.is_eos or self._interim):
                    await self._send_json(
                        {
                            'type': 'transcription',
                            'is_final': ev.is_eos,
                            'alternatives': [{'transcript': text, 'confidence': CONFIDENCE}],
                            'channel': 1,
                            'language': self._language,
                        }
                    )
                if answers_finalize:
                    self._flushed.set()
        finally:
            self._flushed.set()

    async def _send_json(self, payload: dict) -> None:
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await self._ws.send_text(json.dumps(payload))

    async def _teardown(self) -> None:
        if self._forwarder is not None:
            self._forwarder.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._forwarder
        if self._session is not None:
            await self._session.close()
        self._session = None


def authorized(settings: Settings, auth_header: str | None) -> bool:
    return not settings.jambonz_api_key or auth_header == f'Bearer {settings.jambonz_api_key}'


async def handle_stt(ws: WebSocket, settings: Settings, client_factory: Callable[[str | None], Palabra]) -> None:
    if not authorized(settings, ws.headers.get('authorization')):
        log.warning('stt: rejected connection with bad Authorization')
        await ws.close(code=1008, reason='unauthorized')
        return
    await ws.accept()
    await SttConnection(ws, settings, client_factory).run()
