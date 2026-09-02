"""Console entrypoint: ``palabra-jambonz-bridge [--host 0.0.0.0] [--port 8081]``."""

from __future__ import annotations

import argparse
import logging

import uvicorn

from .settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description='jambonz custom speech <-> Palabra Realtime STT/TTS bridge')
    parser.add_argument('--host', default='0.0.0.0')  # noqa: S104 — a server binds all interfaces by design
    parser.add_argument('--port', type=int, default=8081)
    parser.add_argument('--log-level', default=None, help='overrides $LOG_LEVEL (default INFO)')
    args = parser.parse_args()
    settings = Settings.from_env()
    level = (args.log_level or settings.log_level).upper()
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    if not settings.palabra_api_key:
        parser.exit(2, 'PALABRA_API_KEY is not set (create one at https://platform.palabra.ai/api-keys)\n')
    uvicorn.run('palabra_jambonz.app:app', host=args.host, port=args.port, log_level=level.lower())


if __name__ == '__main__':
    main()
