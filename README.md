# Palabra integrations

Integrations of the [Palabra Realtime Speech-to-Text and Text-to-Speech APIs](https://platform.palabra.ai/docs)
with third-party voice platforms and libraries. Every integration is its own PyPI package with its own
README, tests and (for the servers) Docker image.

| Directory | Package | What it is |
|---|---|---|
| [`openai-agents/`](openai-agents/) | `palabra-ai-openai-agents` | `STTModel` / `TTSModel` / `VoiceModelProvider` for the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) `VoicePipeline` |
| [`vapi/`](vapi/) | `palabra-ai-vapi-bridge` | Custom Transcriber (WebSocket) and Custom Voice (HTTP) server for [Vapi](https://vapi.ai) |
| [`jambonz/`](jambonz/) | `palabra-ai-jambonz-bridge` | Custom speech vendor (STT WebSocket + TTS HTTP) for [jambonz](https://jambonz.org) |
| [`wyoming/`](wyoming/) | `wyoming-palabra` | [Wyoming protocol](https://github.com/rhasspy/wyoming) ASR + TTS server for Home Assistant / Rhasspy |

## Common ground

- All integrations use the [`palabra-ai`](https://pypi.org/project/palabra-ai/) client and an API Key from
  [platform.palabra.ai/api-keys](https://platform.palabra.ai/api-keys), passed as `PALABRA_API_KEY`
  (`PALABRA_REGION=us` selects the US endpoints).
- **Audio is never resampled on our side.** Whatever rate the platform delivers (8 kHz telephony, 16 kHz,
  24 kHz, 44.1/48 kHz) is sent to the STT endpoint as the `sample_rate` parameter and resampled server-side.
- **End of speech is signalled with `finalize`.** The recognizer answers with the final transcript marked
  `<fin>`; integrations wait for that marker (and strip it) instead of guessing with timeouts.
- Audio is paced to real time: the realtime recognizer works at wall-clock speed, and frames pushed much
  faster than that are not transcribed. Live sources are never delayed by the pacer.
- TTS text is split into sentences under the server's 1024-character limit and the audio is streamed in
  order, one sentence at a time.
- One API Key supports **one live STT session** at a time (a second connection gets `409`). Servers that
  handle concurrent calls need a pool of keys.

## Development

Python 3.10+.

```bash
make install        # .venv with every package in editable mode + dev extras
make lint           # ruff check + format --check (shared config in ruff.toml)
make test           # offline unit tests (scripted fake of the Palabra client)
PALABRA_API_KEY=plbr_... make test-live   # live tests against the API, one package at a time
make build          # wheels + sdists
make docker         # bridge images
```

The live tests are self-contained: speech for the STT checks is produced by Palabra TTS. GitHub Actions
runs lint, unit tests (Python 3.10–3.13), package builds and Docker builds on every push; live tests run
on `workflow_dispatch` when the `PALABRA_API_KEY` secret is configured.

`docker compose up --build` starts the three bridge servers from `.env` (see `.env.example`).

## Publishing

```bash
cd <integration>
python -m build
twine upload dist/*
```

## License

[MIT](LICENSE)
