# Changelog

## 0.2.0 — 2026-09-02

### All integrations

- **No more client-side resampling.** Audio is sent to Palabra STT at its native
  sample rate via the `sample_rate` parameter; the ASR resamples server-side
  (8 kHz telephony, 16 kHz, 24 kHz Agents SDK audio, 44.1/48 kHz all verified
  against the live API).
- **`finalize` instead of silence padding.** End of input is signalled with the
  `finalize` command; the final transcript answering it (marked `<fin>`) is
  awaited and the marker stripped. The old "append two seconds of silence and
  wait" hack is gone (requires the Palabra STT release of 2026-09-02 for
  sample rates other than 16 kHz).
- Real-time pacing guard for every audio path; sentence-aware splitting of TTS
  text under the 1024-character limit with in-order streaming of the audio.
- Structured logging, injectable Palabra client, `Settings` dataclasses read
  from the environment, `py.typed`.
- Offline unit tests with a scripted fake client plus self-contained live tests
  (speech for the STT tests is produced by Palabra TTS). GitHub Actions CI,
  `docker-compose.yml`, multi-stage non-root Dockerfiles with health checks.

### openai-agents

- `PalabraVoiceModelProvider` for `VoicePipelineConfig(model_provider=...)`.
- `turn_gap_s` option to merge consecutive finals into one turn.
- `TTSModelSettings.speed` honoured (clamped to the Palabra range).
- Examples: microphone agent and a file-in / file-out agent.

### vapi

- Interim transcripts forwarded as `transcriptType: "partial"` when `VAPI_SEND_PARTIALS=true`.
- Request validation and JSON errors for `/tts`; the first audio chunk is
  fetched before the `200` is committed.

### jambonz

- `interimResults` honoured; `stop` triggers `finalize` and the final transcript
  is delivered before the socket is closed, as the protocol requires.
- `{"type": "error"}` messages; configurable TTS sample rate.

### wyoming

- Audio is streamed to Palabra while Home Assistant records (no buffering until
  `audio-stop`), so the transcript arrives right after the user stops speaking.
- Streaming synthesis (`synthesize-start` / `synthesize-chunk` / `synthesize-stop`)
  for Home Assistant's streaming TTS.
- Any sample width / channel count from Wyoming clients is converted to pcm_s16le mono.

## 0.1.0

- Initial release.
