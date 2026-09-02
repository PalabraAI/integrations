"""
Live test of the Vapi bridge against the Palabra API (skipped without PALABRA_API_KEY).

Plays the part of Vapi: synthesizes a phrase through the bridge's own ``/tts``,
then streams it as the customer channel of a stereo 16 kHz call into
``/transcriber`` and checks the final transcript.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from palabra_vapi import Settings, create_app

pytestmark = pytest.mark.live

PHRASE = 'The quick brown fox jumps over the lazy dog near the river bank.'
KEYWORDS = ('quick', 'fox', 'lazy', 'dog', 'river')


@pytest.fixture(scope='module')
def client():
    settings = Settings(palabra_api_key=os.environ['PALABRA_API_KEY'], vapi_secret='live-secret', stt_language='en')
    with TestClient(create_app(settings)) as c:
        yield c


def test_tts_then_transcriber_roundtrip(client):
    r = client.post(
        '/tts',
        headers={'x-vapi-secret': 'live-secret'},
        json={'message': {'type': 'voice-request', 'text': PHRASE, 'sampleRate': 16000}},
    )
    assert r.status_code == 200, r.text
    pcm = np.frombuffer(r.content, dtype=np.int16)
    assert len(pcm) > 16000, 'less than a second of audio'
    print(f'tts: {len(pcm) / 16000:.1f}s @16k')

    stereo = np.column_stack([pcm, np.zeros_like(pcm)]).ravel()  # customer left, silent assistant right
    step = int(16000 * 0.32) * 2
    responses = []
    with client.websocket_connect('/transcriber', headers={'x-vapi-secret': 'live-secret'}) as ws:
        ws.send_text(
            json.dumps(
                {'type': 'start', 'encoding': 'linear16', 'container': 'raw', 'sampleRate': 16000, 'channels': 2}
            )
        )
        for i in range(0, len(stereo), step):
            ws.send_bytes(stereo[i : i + step].tobytes())
            time.sleep(0.32)
        for _ in range(4):  # the call goes on for a bit after the phrase
            ws.send_bytes(np.zeros(step, dtype=np.int16).tobytes())
            time.sleep(0.32)
        ws.send_text(json.dumps({'type': 'stop'}))  # not part of the Vapi protocol; ignored by the bridge
        deadline = time.time() + 8
        while time.time() < deadline:
            try:
                responses.append(ws.receive_json())
                break
            except Exception:
                time.sleep(0.2)
    print('transcriber responses:', responses)
    finals = [m['transcription'].lower() for m in responses if m.get('transcriptType') == 'final']
    assert finals, responses
    assert all(m['channel'] == 'customer' for m in responses)
    assert sum(k in ' '.join(finals) for k in KEYWORDS) >= 3, finals


def test_tts_rejects_bad_secret(client):
    r = client.post('/tts', json={'message': {'type': 'voice-request', 'text': 'x', 'sampleRate': 24000}})
    assert r.status_code == 401
