"""
Vapi Custom Voice (HTTP) backed by Palabra Realtime TTS.

Protocol (https://docs.vapi.ai/customization/custom-voices/custom-tts): Vapi
POSTs ``{"message": {"type": "voice-request", "text": ..., "sampleRate": ...}}``
and expects a ``200`` whose body is raw pcm_s16le mono at exactly that sample
rate — no WAV header — streamed with chunked transfer encoding.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from palabra_ai import Palabra, PalabraError

from ._palabra import short_lang, synthesize
from .settings import Settings

log = logging.getLogger(__name__)

#: Sample rates Vapi may ask for (all supported by Palabra TTS, which does 8-48 kHz).
SAMPLE_RATES = frozenset({8000, 16000, 22050, 24000})


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({'error': message}, status_code=status)


async def handle_voice_request(
    request: Request, settings: Settings, client_factory: Callable[[str | None], Palabra]
) -> Response:
    if settings.vapi_secret and request.headers.get('x-vapi-secret') != settings.vapi_secret:
        log.warning('tts: rejected request with bad x-vapi-secret')
        return _error(401, 'invalid x-vapi-secret')
    try:
        body = await request.json()
    except ValueError:
        return _error(400, 'body must be JSON')
    message = body.get('message', body) if isinstance(body, dict) else None
    if not isinstance(message, dict) or message.get('type', 'voice-request') != 'voice-request':
        return _error(400, 'expected a voice-request message')
    text = str(message.get('text') or '').strip()
    if not text:
        return _error(400, 'text is required')
    try:
        sample_rate = int(message.get('sampleRate', 24000))
    except (TypeError, ValueError):
        return _error(400, 'sampleRate must be an integer')
    if sample_rate not in SAMPLE_RATES:
        return _error(400, f'sampleRate must be one of {sorted(SAMPLE_RATES)}')

    audio = synthesize(
        client_factory(settings.palabra_api_key),
        text,
        language=short_lang(settings.tts_language) or 'en',
        voice_id=settings.voice_id,
        sample_rate=sample_rate,
    )
    # fetch the first chunk before committing to a 200: errors can still be reported as JSON
    try:
        first = await anext(audio)
    except StopAsyncIteration:
        return Response(content=b'', media_type='application/octet-stream')
    except PalabraError as e:
        log.error('tts failed: %s', e)
        return _error(502, f'Palabra TTS failed: {e}')

    async def body_stream() -> AsyncIterator[bytes]:
        yield first
        try:
            async for chunk in audio:
                yield chunk
        except PalabraError as e:  # mid-stream: nothing to do but truncate and log
            log.error('tts stream interrupted: %s', e)

    log.info('tts: %d chars -> pcm_s16le @ %d Hz', len(text), sample_rate)
    return StreamingResponse(body_stream(), media_type='application/octet-stream')
