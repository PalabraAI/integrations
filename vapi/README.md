# Palabra × Vapi

A bridge between [Vapi](https://docs.vapi.ai) and Palabra Realtime STT/TTS, implementing Vapi's
[Custom Transcriber](https://docs.vapi.ai/customization/custom-transcriber) and
[Custom Voice](https://docs.vapi.ai/customization/custom-voices/custom-tts) server protocols.

- **`/transcriber`** (WebSocket) — Vapi streams the call as interleaved stereo linear16 (channel 0 =
  customer, channel 1 = assistant). The bridge splits the channels, feeds each to its own Palabra STT
  session at the call's sample rate (no resampling) and answers with `transcriber-response` messages
  (`transcriptType: "final"`, optionally `"partial"`).
- **`/tts`** (HTTP POST) — Vapi sends a `voice-request`; the bridge streams back raw pcm_s16le mono at the
  requested `sampleRate` (8000 / 16000 / 22050 / 24000), chunked, no WAV header.
- **`/healthz`** (GET).

## Run

```bash
pip install palabra-ai-vapi-bridge
export PALABRA_API_KEY=plbr_...
export VAPI_SECRET=...                  # recommended: enforced as the x-vapi-secret header
palabra-vapi-bridge --port 8080
```

```bash
docker run -e PALABRA_API_KEY=plbr_... -e VAPI_SECRET=... -p 8080:8080 palabra-vapi-bridge
```

The server must be reachable from Vapi (public `https://` / `wss://`, e.g. `ngrok http 8080`).

| Variable | Default | Meaning |
|---|---|---|
| `PALABRA_API_KEY` | — | Palabra API Key (both STT channels and TTS) |
| `VAPI_SECRET` | — | if set, requests without a matching `x-vapi-secret` are rejected |
| `PALABRA_STT_LANGUAGE` | auto-detect | STT source language (`en`, `de-DE`, ...) |
| `VAPI_SEND_PARTIALS` | `false` | also forward interim transcripts as `transcriptType: "partial"` |
| `PALABRA_TTS_LANGUAGE` | `en` | TTS language |
| `PALABRA_VOICE_ID` | `default_low` | TTS voice ([catalog](https://platform.palabra.ai/docs)) |
| `PALABRA_FINALIZE_TIMEOUT` | `10` | seconds to wait for the final transcript when the call ends |
| `LOG_LEVEL` | `INFO` | |

## Vapi assistant configuration

```jsonc
{
  "transcriber": {
    "provider": "custom-transcriber",
    "server": { "url": "wss://<your-host>/transcriber", "secret": "<VAPI_SECRET>" }
  },
  "voice": {
    "provider": "custom-voice",
    "server": { "url": "https://<your-host>/tts", "secret": "<VAPI_SECRET>", "timeoutSeconds": 45 }
  }
}
```

`server.secret` arrives as the `x-vapi-secret` header. Vapi's newer `credentialId` (Bearer / OAuth2 / HMAC
custom credentials) is not verified by the bridge; put it behind a gateway that does, or extend
`handle_transcriber` / `handle_voice_request`.

## Protocol details

**Transcriber.** Vapi sends `{"type": "start", "encoding": "linear16", "container": "raw", "sampleRate": 16000, "channels": 2}`
and then binary frames. Only `linear16` is supported (anything else closes the socket with `1003`). The
bridge replies with

```json
{ "type": "transcriber-response", "transcription": "...", "channel": "customer", "transcriptType": "final" }
```

When the socket closes the bridge flushes both recognizers (`finalize`) so the last words of the call are
still delivered where possible, then closes the Palabra sessions.

**Voice.** `POST /tts` with `{"message": {"type": "voice-request", "text": "...", "sampleRate": 24000}}` →
`200`, `Content-Type: application/octet-stream`, chunked raw PCM. Validation problems and Palabra errors
that happen before the first audio chunk are reported as JSON `{"error": "..."}` with `400` / `401` / `502`.

## Concurrency

Every call opens one Palabra STT session per channel (customer and assistant), all with the same
`PALABRA_API_KEY`. A key is not limited to one live session, so a single bridge instance serves concurrent
calls; the only cap is the concurrent-session quota of your Palabra account.

## Tests

```bash
pip install -e ".[dev]"
pytest -m "not live"                          # offline (fake Palabra client, real FastAPI app)
PALABRA_API_KEY=plbr_... pytest -m live -s    # /tts -> /transcriber round trip against the API
```

## Status

Protocol checked against the current Vapi documentation and exercised end-to-end against the Palabra API
with a test client playing Vapi's part. Not yet run against a live Vapi call.
