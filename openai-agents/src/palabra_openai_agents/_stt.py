"""Palabra Realtime STT as an OpenAI Agents SDK ``STTModel``."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

from agents.voice import AudioInput, StreamedAudioInput, StreamedTranscriptionSession, STTModel, STTModelSettings
from palabra_ai import Palabra, SttSession, SttTranscript

from ._audio import to_pcm_s16le
from ._palabra import Pacer, collect_finals, finalize, short_lang, strip_fin

log = logging.getLogger(__name__)

#: How long ``transcribe()`` waits for the final answering ``finalize``.
DEFAULT_FINALIZE_TIMEOUT_S = 10.0
#: ``StreamedAudioInput`` sample rate by Agents SDK convention.
DEFAULT_INPUT_FRAME_RATE = 24000


class PalabraSTTModel(STTModel):
    """
    Speech-to-text through the Palabra Realtime ASR WebSocket API.

    Args:
        language: source language code (``"en"``, ``"de"``, ...). ``None`` lets
            the server auto-detect it. ``STTModelSettings.language`` overrides it.
        api_key: Palabra API Key; defaults to ``$PALABRA_API_KEY``.
        client: a pre-configured :class:`palabra_ai.Palabra` (region etc.).
        input_frame_rate: sample rate of audio pushed into ``StreamedAudioInput``
            (the SDK convention is 24 kHz). ``AudioInput`` carries its own rate.
        turn_gap_s: for streamed sessions, finals arriving within this many
            seconds of each other are merged into one turn. ``0`` (default)
            yields every final transcript as its own turn.
        finalize_timeout_s: how long to wait for the recognizer to flush after
            the audio ends.

    """

    def __init__(
        self,
        language: str | None = None,
        *,
        api_key: str | None = None,
        client: Palabra | None = None,
        input_frame_rate: int = DEFAULT_INPUT_FRAME_RATE,
        turn_gap_s: float = 0.0,
        finalize_timeout_s: float = DEFAULT_FINALIZE_TIMEOUT_S,
    ):
        self._client = client or Palabra(api_key=api_key)
        self._language = short_lang(language)
        self._input_frame_rate = input_frame_rate
        self._turn_gap_s = turn_gap_s
        self._finalize_timeout_s = finalize_timeout_s

    @property
    def model_name(self) -> str:
        return 'palabra-stt'

    def _language_for(self, settings: STTModelSettings) -> str | None:
        return short_lang(settings.language) or self._language

    async def transcribe(
        self,
        input: AudioInput,
        settings: STTModelSettings,
        trace_include_sensitive_data: bool,
        trace_include_sensitive_audio_data: bool,
    ) -> str:
        """
        Transcribe one static buffer (a user turn).

        The audio is streamed at its native sample rate and paced to real time
        (the realtime recognizer works at wall-clock speed), then ``finalize``
        flushes the tail — so this takes about as long as the audio itself.
        """
        pcm = to_pcm_s16le(input.buffer)
        if not pcm:
            return ''
        async with self._client.stt(language=self._language_for(settings), sample_rate=input.frame_rate) as session:
            await session.send_pcm(pcm, realtime=True)
            await finalize(session)
            finals = await collect_finals(session, timeout=self._finalize_timeout_s)
        return ' '.join(finals)

    async def create_session(
        self,
        input: StreamedAudioInput,
        settings: STTModelSettings,
        trace_include_sensitive_data: bool,
        trace_include_sensitive_audio_data: bool,
    ) -> StreamedTranscriptionSession:
        session = self._client.stt(language=self._language_for(settings), sample_rate=self._input_frame_rate)
        await session.__aenter__()
        return PalabraTranscriptionSession(
            session, input, input_frame_rate=self._input_frame_rate, turn_gap_s=self._turn_gap_s
        )


class PalabraTranscriptionSession(StreamedTranscriptionSession):
    """Pumps a ``StreamedAudioInput`` into one Palabra STT session; final transcripts become turns."""

    def __init__(self, session: SttSession, input: StreamedAudioInput, *, input_frame_rate: int, turn_gap_s: float):
        self._session = session
        self._input = input
        self._input_frame_rate = input_frame_rate
        self._pacer = Pacer(input_frame_rate)
        self._turn_gap_s = turn_gap_s
        self._closed = asyncio.Event()
        self._pump_error: BaseException | None = None
        self._pump_task = asyncio.create_task(self._pump(), name='palabra-stt-pump')

    async def _pump(self) -> None:
        try:
            while True:
                audio = await self._input.queue.get()
                if audio is None:  # end of stream: flush whatever the recognizer still holds
                    await finalize(self._session)
                    return
                pcm = to_pcm_s16le(audio)
                await self._pacer.wait(len(pcm))
                await self._session.send_audio(pcm)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception('Palabra STT audio pump failed')
            self._pump_error = e

    async def transcribe_turns(self) -> AsyncIterator[str]:
        """Yield turns until :meth:`close` is called (or the connection ends)."""
        pending: list[str] = []
        while not self._closed.is_set():
            if self._pump_error is not None:
                raise self._pump_error
            timeout = self._turn_gap_s if pending and self._turn_gap_s > 0 else 0.25
            try:
                ev = await self._session.receive(timeout=timeout)
            except asyncio.TimeoutError:
                if pending and self._turn_gap_s > 0:
                    yield ' '.join(pending)
                    pending = []
                continue
            if ev is None:
                log.info('Palabra STT connection closed by the server')
                break
            if not isinstance(ev, SttTranscript) or not ev.is_eos or ev.is_translation:
                continue
            text = strip_fin(ev.text)
            if not text:
                continue
            if self._turn_gap_s > 0:
                pending.append(text)
            else:
                yield text
        if pending:
            yield ' '.join(pending)

    async def close(self) -> None:
        self._closed.set()
        self._pump_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._pump_task
        await self._session.close()
