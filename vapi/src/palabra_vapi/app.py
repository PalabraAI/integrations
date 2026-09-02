"""FastAPI application: ``/transcriber`` (ws), ``/tts`` (POST), ``/healthz`` (GET)."""

from __future__ import annotations

import logging
from collections.abc import Callable

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response
from palabra_ai import Palabra

from . import __version__
from .settings import Settings
from .transcriber import handle_transcriber
from .voice import handle_voice_request

log = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    client_factory: Callable[[str | None], Palabra] | None = None,
) -> FastAPI:
    """Build the bridge. ``client_factory(api_key)`` is injectable for tests."""
    settings = settings or Settings.from_env()
    factory = client_factory or (lambda key: Palabra(api_key=key))
    if not settings.palabra_api_key and client_factory is None:
        log.warning('PALABRA_API_KEY is not set: every Palabra call will fail with 401')

    app = FastAPI(title='palabra-vapi-bridge', version=__version__, docs_url=None, redoc_url=None)
    app.state.settings = settings

    @app.get('/healthz')
    async def healthz() -> dict:
        return {'status': 'ok', 'version': __version__}

    @app.websocket('/transcriber')
    async def transcriber(ws: WebSocket) -> None:
        await handle_transcriber(ws, settings, factory)

    @app.post('/tts')
    async def tts(request: Request) -> Response:
        return await handle_voice_request(request, settings, factory)

    return app


app = create_app()
