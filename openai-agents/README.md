# Palabra × OpenAI Agents SDK

Palabra Realtime STT and TTS as models for the [`VoicePipeline`](https://openai.github.io/openai-agents-python/voice/pipeline/)
of the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python). Speech recognition and synthesis go
through Palabra instead of `gpt-4o-transcribe` / `tts-1`; the agent (LLM, tools, handoffs) stays as it is.

```bash
pip install palabra-ai-openai-agents
export PALABRA_API_KEY=plbr_...   # https://platform.palabra.ai/api-keys
```

## Usage

```python
from agents import Agent
from agents.voice import SingleAgentVoiceWorkflow, VoicePipeline
from palabra_openai_agents import PalabraSTTModel, PalabraTTSModel

agent = Agent(name="Assistant", instructions="You are a helpful voice assistant.")

pipeline = VoicePipeline(
    workflow=SingleAgentVoiceWorkflow(agent),
    stt_model=PalabraSTTModel(language="en"),                 # None = server-side language detection
    tts_model=PalabraTTSModel(language="en", voice_id="default_low"),
)
```

Then use the pipeline as usual: `await pipeline.run(AudioInput(buffer=..., frame_rate=...))` for a
recorded turn, or `StreamedAudioInput` for a live microphone. Or hand out both models through the config:

```python
from agents.voice import VoicePipelineConfig
from palabra_openai_agents import PalabraVoiceModelProvider

config = VoicePipelineConfig(model_provider=PalabraVoiceModelProvider(stt_language="en", tts_language="en"))
pipeline = VoicePipeline(workflow=..., config=config)
```

Examples: [`examples/voice_agent_mic.py`](examples/voice_agent_mic.py) (microphone → agent → speakers,
needs `sounddevice` and `OPENAI_API_KEY`) and [`examples/voice_agent_file.py`](examples/voice_agent_file.py)
(WAV question in, WAV answer out).

## How it maps

| SDK interface | Implementation |
|---|---|
| `STTModel.transcribe(AudioInput)` | the buffer (int16 or float32, any rate) is streamed at its native `frame_rate`, paced to real time; then `finalize` is sent and the finals up to the `<fin>` marker are joined. Takes about as long as the audio. |
| `STTModel.create_session(StreamedAudioInput)` | a pump task feeds queued chunks (24 kHz by SDK convention, `input_frame_rate=` to change) into one STT WebSocket; `transcribe_turns()` yields every final transcript as a turn (`turn_gap_s=` merges finals that arrive close together). `add_audio(None)` ends the input with `finalize`. |
| `TTSModel.run(text)` | one TTS WebSocket per call, `pcm` s16le mono at 24 kHz — the format `VoicePipeline` expects. Text is split into sentences under the 1024-character limit and the audio streamed in order. `TTSModelSettings.speed` is honoured (clamped to 0.1–2.0); `settings.voice` (OpenAI names) is ignored — pick a Palabra voice with `voice_id=`. |

Options of `PalabraSTTModel`: `language`, `input_frame_rate`, `turn_gap_s`, `finalize_timeout_s`,
`api_key` / `client` (a configured `palabra_ai.Palabra`, e.g. for `region="us"`).

## Tests

```bash
pip install -e ".[dev]"
pytest -m "not live"                          # offline, scripted fake of the Palabra client
PALABRA_API_KEY=plbr_... pytest -m live -s    # against the API; speech is produced by Palabra TTS
PALABRA_TEST_WAV=my.wav PALABRA_TEST_LANGUAGE=ru pytest -m live -k user_recording -s
```

## Notes

- One API Key is not limited to a single live STT session: parallel `create_session` / `transcribe` calls
  each open their own session with the same key.
