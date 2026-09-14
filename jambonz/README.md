# Palabra × jambonz

A [jambonz custom speech](https://docs.jambonz.org/guides/features/custom-stt-providers) vendor backed by
Palabra Realtime STT/TTS: plug Palabra into the jambonz CPaaS as the recognizer and synthesizer of your
applications.

- **`/stt`** (WebSocket, no subprotocol) — jambonz sends `start` (`language`, `format: "raw"`,
  `encoding: "LINEAR16"`, `sampleRateHz`, `interimResults`), binary linear16 frames, then `stop`. Audio is
  forwarded to Palabra at the call's sample rate (8 kHz telephony as a rule — no resampling); Palabra
  transcripts come back as `transcription` messages (interim ones only when `interimResults` is on). On
  `stop` the bridge sends `finalize`, delivers the final transcript and closes the socket, as the protocol
  requires.
- **`/tts`** (HTTP POST) — jambonz sends `{language, voice, type: "text"|"ssml", text}`; the bridge returns
  the whole utterance as `audio/wav` (SSML tags are stripped — Palabra TTS takes plain text).
- **`/healthz`** (GET).

## Run

```bash
pip install palabra-ai-jambonz-bridge
export PALABRA_API_KEY=plbr_...
export JAMBONZ_API_KEY=...          # the API key you enter for the vendor in the jambonz portal
palabra-jambonz-bridge --port 8081
```

```bash
docker run -e PALABRA_API_KEY=plbr_... -e JAMBONZ_API_KEY=... -p 8081:8081 palabra-jambonz-bridge
```

| Variable | Default | Meaning |
|---|---|---|
| `PALABRA_API_KEY` | — | Palabra API Key |
| `JAMBONZ_API_KEY` | — | if set, `Authorization: Bearer <key>` is required on `/stt` and `/tts` |
| `PALABRA_VOICE_ID` | `default_low` | TTS voice when jambonz sends none |
| `PALABRA_TTS_SAMPLE_RATE` | `24000` | sample rate of the returned WAV (8000–48000) |
| `PALABRA_FINALIZE_TIMEOUT` | `10` | seconds to wait for the final transcript after `stop` |
| `LOG_LEVEL` | `INFO` | |

## jambonz configuration

Portal: **Speech → Add speech service → vendor "Custom"** → name, API key (the same value as
`JAMBONZ_API_KEY`), STT URL `wss://<your-host>/stt`, TTS URL `https://<your-host>/tts`. Applications then
use `custom:<name>` as the recognizer / synthesizer vendor.

## Protocol details

Transcription messages:

```json
{ "type": "transcription", "is_final": true,
  "alternatives": [{ "transcript": "...", "confidence": 0.9 }],
  "channel": 1, "language": "en-US" }
```

`language` echoes the BCP-47 code jambonz sent; Palabra receives the short code (`en-US` → `en`).
Palabra does not report confidence, so a constant `0.9` is returned. Errors are
`{"type": "error", "error": "..."}`. `options.hints` are accepted and ignored. jambonz's separate
streaming-TTS WebSocket protocol (`connect` / `stream` / `flush` / `stop`) is not implemented.

## Tests

```bash
pip install -e ".[dev]"
pytest -m "not live"                          # offline (fake Palabra client, real FastAPI app)
PALABRA_API_KEY=plbr_... pytest -m live -s    # /tts (8 kHz wav) -> /stt round trip against the API
```

## Status

Protocol checked against the current jambonz documentation and the reference
[custom-speech-example](https://github.com/jambonz/custom-speech-example); exercised end-to-end against the
Palabra API with a test client playing jambonz's part (8 kHz). Not yet run against a live jambonz.
Concurrent calls each open their own Palabra STT session with the same API Key.
