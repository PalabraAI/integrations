"""Offline tests of the Vapi bridge with a scripted fake Palabra client."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from palabra_vapi import Settings, create_app

from .conftest import FakePalabra, transcript


@pytest.fixture
def fake() -> FakePalabra:
    return FakePalabra()


def make_client(fake: FakePalabra, **overrides) -> TestClient:
    settings = Settings(palabra_api_key='k-customer', **overrides)
    return TestClient(create_app(settings, client_factory=fake.for_key))


def test_healthz(fake):
    with make_client(fake) as client:
        r = client.get('/healthz')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'


def test_settings_from_env():
    env = {
        'PALABRA_API_KEY': 'k',
        'VAPI_SECRET': 's',
        'PALABRA_LANGUAGE': 'de-DE',
        'VAPI_SEND_PARTIALS': 'true',
        'PALABRA_VOICE_ID': 'default_high',
    }
    s = Settings.from_env(env)
    assert (s.palabra_api_key, s.vapi_secret) == ('k', 's')
    assert s.stt_language == 'de-DE'
    assert s.send_partials is True
    assert s.voice_id == 'default_high'


def test_transcriber_rejects_bad_secret(fake):
    with make_client(fake, vapi_secret='shh') as client, pytest.raises(Exception):  # noqa: B017 — starlette raises its own
        with client.websocket_connect('/transcriber', headers={'x-vapi-secret': 'nope'}):
            pass


def test_transcriber_splits_stereo_and_forwards_finals_at_native_rate():
    fake = FakePalabra(stt_script=[transcript('partial', eos=False), transcript('Hello from the customer.')])
    with make_client(fake, vapi_secret='shh') as client:
        with client.websocket_connect('/transcriber', headers={'x-vapi-secret': 'shh'}) as ws:
            ws.send_text(
                json.dumps(
                    {'type': 'start', 'encoding': 'linear16', 'container': 'raw', 'sampleRate': 8000, 'channels': 2}
                )
            )
            left = np.full(800, 1000, dtype=np.int16)
            right = np.full(800, -1000, dtype=np.int16)
            ws.send_bytes(np.column_stack([left, right]).ravel().tobytes())
            first = ws.receive_json()
            second = ws.receive_json()
    got = sorted((m['channel'], m['transcription'], m['transcriptType']) for m in (first, second))
    assert got == [
        ('assistant', 'Hello from the customer.', 'final'),
        ('customer', 'Hello from the customer.', 'final'),
    ]
    customer, assistant = fake.stt_sessions  # one session per channel, both on the single API Key
    assert customer.params == {'language': None, 'sample_rate': 8000}
    assert customer.sent == [left.tobytes()]  # de-interleaved: no resampling, no padding
    assert assistant.sent == [right.tobytes()]
    # on disconnect the bridge finalized both channels
    assert customer.finalized == 1
    assert assistant.finalized == 1
    assert customer.closed
    assert assistant.closed


def test_transcriber_mono_call_and_partials():
    fake = FakePalabra(stt_script=[transcript('Hel', eos=False), transcript('Hello.')])
    with make_client(fake, send_partials=True, stt_language='en-US') as client:
        with client.websocket_connect('/transcriber') as ws:
            ws.send_text(json.dumps({'type': 'start', 'sampleRate': 16000, 'channels': 1}))
            ws.send_bytes(np.zeros(640, dtype=np.int16).tobytes())
            first = ws.receive_json()
            second = ws.receive_json()
    assert first == {
        'type': 'transcriber-response',
        'transcription': 'Hel',
        'channel': 'customer',
        'transcriptType': 'partial',
    }
    assert second['transcriptType'] == 'final'
    assert len(fake.stt_sessions) == 1  # a mono call has no assistant channel
    assert fake.stt_sessions[0].params['language'] == 'en'


def test_tts_streams_raw_pcm_at_requested_rate(fake):
    fake.for_key = lambda _key: fake
    with make_client(fake) as client:
        r = client.post(
            '/tts', json={'message': {'type': 'voice-request', 'text': 'Hello there. Bye now.', 'sampleRate': 8000}}
        )
    assert r.status_code == 200
    assert r.headers['content-type'] == 'application/octet-stream'
    assert r.content == b'Hello there.Bye now.'  # the fake echoes text as "audio", one sentence per message
    assert fake.tts_sessions[0].init == {
        'language': 'en',
        'voice_id': 'default_low',
        'speed': None,
        'format': 'pcm',
        'sample_rate': 8000,
    }


@pytest.mark.parametrize(
    ('body', 'status', 'needle'),
    [
        ({'message': {'type': 'voice-request', 'sampleRate': 24000}}, 400, 'text'),
        ({'message': {'type': 'voice-request', 'text': 'hi', 'sampleRate': 11025}}, 400, 'sampleRate'),
        ({'message': {'type': 'something-else', 'text': 'hi'}}, 400, 'voice-request'),
    ],
)
def test_tts_validation(fake, body, status, needle):
    fake.for_key = lambda _key: fake
    with make_client(fake) as client:
        r = client.post('/tts', json=body)
    assert r.status_code == status
    assert needle in r.json()['error']


def test_tts_requires_secret(fake):
    fake.for_key = lambda _key: fake
    with make_client(fake, vapi_secret='shh') as client:
        r = client.post('/tts', json={'message': {'type': 'voice-request', 'text': 'hi', 'sampleRate': 24000}})
    assert r.status_code == 401
