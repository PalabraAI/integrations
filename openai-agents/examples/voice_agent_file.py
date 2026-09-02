"""
Voice agent over files: WAV question -> Palabra STT -> OpenAI agent -> Palabra TTS -> WAV answer.

Handy for CI or for trying the pipeline without a microphone.

    PALABRA_API_KEY=plbr_... OPENAI_API_KEY=sk-... python voice_agent_file.py question.wav answer.wav [--language en]
"""

from __future__ import annotations

import argparse
import asyncio
import wave

import numpy as np
from agents import Agent
from agents.voice import AudioInput, SingleAgentVoiceWorkflow, VoicePipeline

from palabra_openai_agents import PalabraSTTModel, PalabraTTSModel

TTS_RATE = 24000


async def main(question_wav: str, answer_wav: str, language: str) -> None:
    with wave.open(question_wav) as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit('expected a 16-bit mono WAV')
        rate = w.getframerate()
        buffer = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

    agent = Agent(
        name='Assistant', instructions=f'You are a helpful assistant. Answer in one short sentence, in {language}.'
    )
    pipeline = VoicePipeline(
        workflow=SingleAgentVoiceWorkflow(agent),
        stt_model=PalabraSTTModel(language=language),
        tts_model=PalabraTTSModel(language=language),
    )

    # the ASR takes the WAV at its native rate; no resampling on our side
    result = await pipeline.run(AudioInput(buffer=buffer, frame_rate=rate))
    audio = bytearray()
    async for event in result.stream():
        if event.type == 'voice_stream_event_audio':
            audio.extend(event.data.tobytes())
        elif event.type == 'voice_stream_event_error':
            raise SystemExit(f'pipeline error: {event.error}')

    with wave.open(answer_wav, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TTS_RATE)
        w.writeframes(bytes(audio))
    print(f'wrote {answer_wav}: {len(audio) / 2 / TTS_RATE:.1f}s of speech')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('question_wav')
    parser.add_argument('answer_wav')
    parser.add_argument('--language', default='en')
    args = parser.parse_args()
    asyncio.run(main(args.question_wav, args.answer_wav, args.language))
