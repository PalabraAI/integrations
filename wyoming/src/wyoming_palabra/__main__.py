"""Console entrypoint: ``wyoming-palabra --uri tcp://0.0.0.0:10300``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from functools import partial

from palabra_ai import Palabra
from wyoming.server import AsyncServer

from . import __version__
from .handler import HandlerConfig, PalabraEventHandler
from .info import DEFAULT_VOICES, build_info

log = logging.getLogger('wyoming_palabra')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Wyoming protocol server backed by Palabra Realtime STT/TTS')
    parser.add_argument(
        '--uri', default=os.environ.get('WYOMING_URI', 'tcp://0.0.0.0:10300'), help='tcp://host:port or unix://path'
    )
    parser.add_argument(
        '--language', default=os.environ.get('PALABRA_LANGUAGE', 'en'), help='TTS language when the client does not say'
    )
    parser.add_argument(
        '--voice', action='append', default=None, help='voice id to advertise (repeatable; default: built-in voices)'
    )
    parser.add_argument('--default-voice', default=os.environ.get('PALABRA_VOICE_ID', 'default_low'))
    parser.add_argument('--log-level', default=os.environ.get('LOG_LEVEL', 'INFO'))
    parser.add_argument('--version', action='version', version=__version__)
    return parser.parse_args(argv)


async def serve(args: argparse.Namespace) -> None:
    client = Palabra()  # reads PALABRA_API_KEY / PALABRA_REGION
    voices = args.voice or DEFAULT_VOICES
    if args.default_voice not in voices:
        voices = [args.default_voice, *voices]
    config = HandlerConfig(info=build_info(voices), default_language=args.language, default_voice=args.default_voice)
    server = AsyncServer.from_uri(args.uri)
    log.info('wyoming-palabra %s listening on %s (voices: %s)', __version__, args.uri, ', '.join(voices))
    await server.run(partial(PalabraEventHandler, config, client))


def run(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    if not os.environ.get('PALABRA_API_KEY'):
        raise SystemExit('PALABRA_API_KEY is not set (create one at https://platform.palabra.ai/api-keys)')
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    run()
