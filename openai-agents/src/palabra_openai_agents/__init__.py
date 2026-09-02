"""
Palabra STT/TTS models for the OpenAI Agents SDK ``VoicePipeline``.

Drop-in replacements for ``OpenAISTTModel`` / ``OpenAITTSModel`` backed by the
Palabra Realtime Speech-to-Text and Text-to-Speech APIs
(https://platform.palabra.ai/docs)::

    from agents.voice import SingleAgentVoiceWorkflow, VoicePipeline
    from palabra_openai_agents import PalabraSTTModel, PalabraTTSModel

    pipeline = VoicePipeline(
        workflow=SingleAgentVoiceWorkflow(agent),
        stt_model=PalabraSTTModel(language="en"),
        tts_model=PalabraTTSModel(language="en", voice_id="default_low"),
    )

Auth: ``PALABRA_API_KEY`` (https://platform.palabra.ai/api-keys), or ``api_key=`` / ``client=``.
"""

from ._provider import PalabraVoiceModelProvider
from ._stt import PalabraSTTModel, PalabraTranscriptionSession
from ._tts import PalabraTTSModel

__version__ = '0.2.0'
__all__ = ['PalabraSTTModel', 'PalabraTTSModel', 'PalabraTranscriptionSession', 'PalabraVoiceModelProvider']
