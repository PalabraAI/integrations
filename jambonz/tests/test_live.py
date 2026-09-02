"""
Live test of the jambonz bridge against the Palabra API (skipped without PALABRA_API_KEY).

Plays the part of jambonz: fetches a WAV from ``/tts`` (8 kHz, like a phone
call), streams it as LINEAR16 @ 8 kHz into ``/stt`` and sends ``stop``.
"""

from __future__ import annotations

import io
import json
import os
import time
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from palabra_jambonz import Settings, create_app

pytestmark = pytest.mark.live

PHRASE = 'Good afternoon, thank you for calling. How can I help you today?'
KEYWORDS = ('afternoon', 'thank', 'calling', 'help', 'today')


@pytest.fixture(scope='module')
def client():
    settings = Settings(palabra_api_key=os.environ['PALABRA_API_KEY'], jambonz_api_key='live-key', tts_sample_rate=8000)
    with TestClient(create_app(settings)) as c:
        yield c


def test_tts_then_stt_roundtrip_at_8k(client):
    r = client.post(
        '/tts',
        headers={'authorization': 'Bearer live-key'},
        json={'language': 'en-US', 'voice': 'default_low', 'type': 'text', 'text': PHRASE},
    )
    assert r.status_code == 200, r.text
    with wave.open(io.BytesIO(r.content)) as w:
        assert w.getframerate() == 8000
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    print(f'tts: {len(pcm) / 8000:.1f}s @8k')

    messages = []
    step = int(8000 * 0.32)
    with client.websocket_connect('/stt', headers={'authorization': 'Bearer live-key'}) as ws:
        ws.send_text(
            json.dumps(
                {
                    'type': 'start',
                    'language': 'en-US',
                    'format': 'raw',
                    'encoding': 'LINEAR16',
                    'sampleRateHz': 8000,
                    'interimResults': False,
                    'options': {},
                }
            )
        )
        for i in range(0, len(pcm), step):
            ws.send_bytes(pcm[i : i + step].tobytes())
            time.sleep(0.32)
        ws.send_text(json.dumps({'type': 'stop'}))
        try:
            while True:
                messages.append(ws.receive_json())
        except Exception:
            pass
    print('stt messages:', messages)
    finals = [m['alternatives'][0]['transcript'].lower() for m in messages if m.get('is_final')]
    assert finals, messages
    assert sum(k in ' '.join(finals) for k in KEYWORDS) >= 3, finals
