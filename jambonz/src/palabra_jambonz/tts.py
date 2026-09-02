"""
jambonz custom TTS vendor (HTTP) backed by Palabra Realtime TTS.

Protocol (https://docs.jambonz.org/guides/features/custom-tts-providers):
jambonz POSTs ``{"language", "voice", "type": "text"|"ssml", "text"}`` with
``Authorization: Bearer <apiKey>`` and expects ``200`` with the whole audio
file; ``audio/wav`` is one of the accepted content types. jambonz caches the
result and transcodes for the call.
"""

from __future__ import annotations

import io
import logging
import re
import wave
from collections.abc import Callable

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from palabra_ai import Palabra, PalabraError

from ._palabra import short_lang, synthesize
from .settings import Settings
from .stt import authorized

log = logging.getLogger(__name__)

_SSML_TAG = re.compile(r'<[^>]+>')
_MAX_TEXT = 20_000  # sanity limit on one request


def _wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


async def handle_tts(request: Request, settings: Settings, client_factory: Callable[[str | None], Palabra]) -> Response:
    if not authorized(settings, request.headers.get('authorization')):
        log.warning('tts: rejected request with bad Authorization')
        return JSONResponse({'error': 'unauthorized'}, status_code=401)
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse({'error': 'body must be JSON'}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({'error': 'body must be a JSON object'}, status_code=400)
    text = str(body.get('text') or '')
    if body.get('type') == 'ssml':
        text = _SSML_TAG.sub(' ', text)  # Palabra TTS takes plain text
    text = ' '.join(text.split())
    if not text:
        return JSONResponse({'error': 'text is required'}, status_code=400)
    if len(text) > _MAX_TEXT:
        return JSONResponse({'error': f'text longer than {_MAX_TEXT} characters'}, status_code=413)
    language = short_lang(body.get('language')) or 'en'
    voice_id = str(body.get('voice') or settings.voice_id)

    pcm = bytearray()
    try:
        async for chunk in synthesize(
            client_factory(settings.palabra_api_key),
            text,
            language=language,
            voice_id=voice_id,
            sample_rate=settings.tts_sample_rate,
        ):
            pcm.extend(chunk)
    except PalabraError as e:
        log.error('tts failed: %s', e)
        return JSONResponse({'error': f'Palabra TTS failed: {e}'}, status_code=502)
    log.info(
        'tts: %d chars (%s, %s) -> %.1fs wav', len(text), language, voice_id, len(pcm) / 2 / settings.tts_sample_rate
    )
    return Response(content=_wav(bytes(pcm), settings.tts_sample_rate), media_type='audio/wav')
