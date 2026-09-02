"""
Voice agent: microphone -> Palabra STT -> OpenAI agent -> Palabra TTS -> speakers.

pip install "palabra-ai-openai-agents[examples]"
PALABRA_API_KEY=plbr_... OPENAI_API_KEY=sk-... python voice_agent_mic.py [--language en]
"""

from __future__ import annotations

import argparse
import asyncio

import numpy as np
import sounddevice as sd
from agents import Agent
from agents.voice import SingleAgentVoiceWorkflow, StreamedAudioInput, VoicePipeline

from palabra_openai_agents import PalabraSTTModel, PalabraTTSModel

SAMPLE_RATE = 24000  # the Agents SDK voice convention (mic in and TTS out)
CHUNK = int(SAMPLE_RATE * 0.32)


async def main(language: str, voice_id: str) -> None:
    agent = Agent(
        name='Assistant',
        instructions=f'You are a helpful voice assistant. Answer briefly, in {language}.',
    )
    pipeline = VoicePipeline(
        workflow=SingleAgentVoiceWorkflow(agent),
        stt_model=PalabraSTTModel(language=language, input_frame_rate=SAMPLE_RATE),
        tts_model=PalabraTTSModel(language=language, voice_id=voice_id),
    )

    audio_input = StreamedAudioInput()
    result = await pipeline.run(audio_input)
    loop = asyncio.get_running_loop()

    def on_mic(indata, frames, time, status) -> None:
        pcm = np.frombuffer(bytes(indata), dtype=np.int16).copy()
        loop.call_soon_threadsafe(audio_input.queue.put_nowait, pcm)

    with (
        sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16') as player,
        sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', blocksize=CHUNK, callback=on_mic),
    ):
        print('Speak! Ctrl+C to stop')
        async for event in result.stream():
            if event.type == 'voice_stream_event_audio':
                player.write(event.data)
            elif event.type == 'voice_stream_event_error':
                print(f'error: {event.error}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--language', default='en', help='language spoken by the user and the agent')
    parser.add_argument('--voice', default='default_low', help='Palabra voice id')
    args = parser.parse_args()
    try:
        asyncio.run(main(args.language, args.voice))
    except KeyboardInterrupt:
        print('\nDone')
