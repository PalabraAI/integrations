"""Audio buffer conversion for the Agents SDK <-> Palabra boundary."""

from __future__ import annotations

import numpy as np


def to_pcm_s16le(buffer: np.ndarray) -> bytes:
    """
    Agents SDK audio buffer (int16 or float32, mono or ``(frames, channels)``) -> pcm_s16le bytes.

    No resampling happens here: the sample rate travels to the ASR endpoint as
    the ``sample_rate`` parameter and the server resamples as needed.
    """
    if buffer.ndim == 2:  # (frames, channels) -> mono, keeping the dtype
        buffer = buffer.mean(axis=1).astype(buffer.dtype)
    elif buffer.ndim != 1:
        raise ValueError(f'audio buffer must be 1-D or (frames, channels), got shape {buffer.shape}')
    if buffer.dtype == np.int16:
        return buffer.tobytes()
    if buffer.dtype in (np.float32, np.float64):
        return (np.clip(buffer, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    raise ValueError(f'audio buffer must be int16 or float32, got {buffer.dtype}')
