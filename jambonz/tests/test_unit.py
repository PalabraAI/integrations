"""Offline tests of the jambonz bridge with a scripted fake Palabra client."""

from __future__ import annotations

import io
import json
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from palabra_jambonz import Settings, create_app

from .conftest import FakePalabra, transcript


def make_client(fake: FakePalabra, **overrides) -> TestClient:
    settings = Settings(palabra_api_key='k', **overrides)
    return TestClient(create_app(settings, client_factory=lambda _key: fake))


def test_healthz():
    with make_client(FakePalabra()) as client:
        assert client.get('/healthz').json()['status'] == 'ok'


def test_settings_from_env():
    s = Settings.from_env({'PALABRA_API_KEY': 'k', 'JAMBONZ_API_KEY': 'j', 'PALABRA_TTS_SAMPLE_RATE': '8000'})
    assert (s.palabra_api_key, s.jambonz_api_key, s.tts_sample_rate) == ('k', 'j', 8000)


def test_stt_requires_bearer_when_configured():
    with make_client(FakePalabra(), jambonz_api_key='secret') as client, pytest.raises(Exception):  # noqa: B017
        with client.websocket_connect('/stt', headers={'authorization': 'Bearer wrong'}):
            pass


def test_stt_start_audio_stop_flow_at_8k():
    fake = FakePalabra(stt_script=[transcript('hel', eos=False), transcript('Hello there.')], fin_text='Goodbye.<fin>')
    with make_client(fake, jambonz_api_key='secret') as client:
        with client.websocket_connect('/stt', headers={'authorization': 'Bearer secret'}) as ws:
            ws.send_text(
                json.dumps(
                    {
                        'type': 'start',
                        'language': 'en-US',
                        'format': 'raw',
                        'encoding': 'LINEAR16',
                        'sampleRateHz': 8000,
                        'interimResults': True,
                        'options': {},
                    }
                )
            )
            audio = np.full(1600, 500, dtype=np.int16).tobytes()
            ws.send_bytes(audio)
            interim = ws.receive_json()
            final = ws.receive_json()
            ws.send_text(json.dumps({'type': 'stop'}))
            after_stop = ws.receive_json()
            with pytest.raises(Exception):  # noqa: B017 — the bridge closes the socket after stop
                ws.receive_json()
    assert interim == {
        'type': 'transcription',
        'is_final': False,
        'alternatives': [{'transcript': 'hel', 'confidence': 0.9}],
        'channel': 1,
        'language': 'en-US',
    }
    assert final['is_final'] is True
    assert final['alternatives'][0]['transcript'] == 'Hello there.'
    assert after_stop['alternatives'][0]['transcript'] == 'Goodbye.'  # <fin> marker stripped
    session = fake.stt_sessions[0]
    assert session.params == {'language': 'en', 'sample_rate': 8000}
    assert session.sent == [audio]  # forwarded as-is: no resampling, no padding
    assert session.finalized == 1
    assert session.closed


def test_stt_interim_off_forwards_only_finals():
    fake = FakePalabra(stt_script=[transcript('hel', eos=False), transcript('Hello.')])
    with make_client(fake) as client, client.websocket_connect('/stt') as ws:
        ws.send_text(json.dumps({'type': 'start', 'language': 'de-DE', 'sampleRateHz': 16000, 'interimResults': False}))
        msg = ws.receive_json()
    assert msg['is_final'] is True
    assert msg['language'] == 'de-DE'
    assert fake.stt_sessions[0].params['language'] == 'de'


def test_stt_unsupported_encoding_reports_error():
    fake = FakePalabra()
    with make_client(fake) as client, client.websocket_connect('/stt') as ws:
        ws.send_text(json.dumps({'type': 'start', 'encoding': 'MULAW', 'sampleRateHz': 8000}))
        msg = ws.receive_json()
    assert msg['type'] == 'error'
    assert 'mulaw' in msg['error']


def test_tts_returns_wav_and_strips_ssml():
    fake = FakePalabra()
    with make_client(fake, jambonz_api_key='secret', tts_sample_rate=8000) as client:
        r = client.post(
            '/tts',
            headers={'authorization': 'Bearer secret'},
            json={
                'language': 'fr-FR',
                'voice': 'default_high',
                'type': 'ssml',
                'text': '<speak>Bonjour <break/> tout le monde.</speak>',
            },
        )
    assert r.status_code == 200
    assert r.headers['content-type'] == 'audio/wav'
    with wave.open(io.BytesIO(r.content)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (8000, 1, 2)
        assert w.readframes(w.getnframes()) == b'Bonjour tout le monde.'
    session = fake.tts_sessions[0]
    assert session.init['language'] == 'fr'
    assert session.init['voice_id'] == 'default_high'
    assert session.init['sample_rate'] == 8000


def test_tts_validation_and_auth():
    fake = FakePalabra()
    with make_client(fake, jambonz_api_key='secret') as client:
        assert client.post('/tts', json={'text': 'x'}).status_code == 401
        r = client.post('/tts', headers={'authorization': 'Bearer secret'}, json={'text': '   '})
        assert r.status_code == 400
