# Palabra × Wyoming (Home Assistant / Rhasspy)

A [Wyoming protocol](https://github.com/rhasspy/wyoming) server exposing Palabra Realtime STT and TTS as a
speech provider for Home Assistant voice assistants (and any other Wyoming client, e.g. Rhasspy 3).

- **ASR** — `transcribe` + `audio-start` / `audio-chunk` / `audio-stop`. Audio is forwarded to Palabra as it
  arrives, at the rate Home Assistant records at (no resampling; other sample widths and channel counts are
  converted to 16-bit mono). `audio-stop` sends `finalize`; the final transcript is returned as `transcript`.
- **TTS** — `synthesize` → `audio-start`, `audio-chunk`… (pcm_s16le 24 kHz mono, streamed as generated),
  `audio-stop`. Home Assistant's streaming synthesis (`synthesize-start` / `synthesize-chunk` /
  `synthesize-stop`) is supported: completed sentences are synthesized while the LLM is still writing.
- **Discovery** — `describe` returns the ASR model (13 languages) and the voices.

## Run

```bash
pip install wyoming-palabra
export PALABRA_API_KEY=plbr_...
wyoming-palabra --uri tcp://0.0.0.0:10300
```

```bash
docker run -e PALABRA_API_KEY=plbr_... -p 10300:10300 wyoming-palabra
```

Options: `--language` (TTS language when the request does not say; default `en`), `--default-voice`
(`default_low`), `--voice <id>` (repeatable — advertise additional voices, e.g. ones cloned through the
Palabra Voices API), `--log-level`. Environment equivalents: `WYOMING_URI`, `PALABRA_LANGUAGE`,
`PALABRA_VOICE_ID`, `LOG_LEVEL`, plus `PALABRA_REGION`.

## Home Assistant

Settings → Devices & services → **Add integration → Wyoming Protocol** → host and port of the server
(10300). Palabra then appears as a Speech-to-text and Text-to-speech engine under Settings → Voice
assistants.

Voices are advertised per language as `<lang>-<voice>` (`en-default_low`, `de-default_high`, …) because a
Wyoming `synthesize` request carries either a voice name or a language, never both. Home Assistant shows
only the voices matching the assistant's language; the server derives the synthesis language from the
chosen voice, falling back to the language of the last `transcribe`, then `--language`.

## Tests

```bash
pip install -e ".[dev]"
pytest -m "not live"                          # offline: a real Wyoming client against the server with a fake Palabra
PALABRA_API_KEY=plbr_... pytest -m live -s    # describe -> synthesize -> transcribe round trip against the API
```

## Status

Checked against wyoming 1.10 (required `Info` / `Artifact` fields, optional `Synthesize.voice`, one handler
per connection) and exercised end-to-end with `wyoming.client.AsyncTcpClient` — the class Home Assistant
uses — against the Palabra API. Not yet run inside a live Home Assistant. Palabra STT allows one live
session per API Key, so one voice command is transcribed at a time.
