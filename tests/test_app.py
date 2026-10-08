import json

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from app.protocol import CLIENT_SAMPLE_RATE


@pytest.fixture
def fake_mode(monkeypatch):
    """Run the app against the deterministic fakes: no network, no credits."""
    get_settings.cache_clear()
    monkeypatch.setenv("USE_FAKES", "true")
    yield
    get_settings.cache_clear()


def test_healthz_reports_mode_and_sample_rate(fake_mode):
    body = TestClient(app).get("/healthz").json()
    assert body["status"] == "ok"
    assert body["mode"] == "fakes"
    assert body["client_sample_rate"] == str(CLIENT_SAMPLE_RATE)


def test_index_page_is_served():
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "KhataCall" in response.text
    assert "Hold to speak" in response.text


def test_settings_defaults_match_documented_gnani_values():
    settings = Settings(_env_file=None)
    assert settings.gnani_base_url == "https://api.vachana.ai"
    assert settings.stt_language == "en-IN"
    assert settings.tts_voice == "Poorvi"  # only hi-en voice in the catalog (D-011)
    assert settings.use_fakes is False


def test_ws_turn_end_to_end_over_the_real_endpoint(fake_mode):
    """Drive /ws exactly as the browser does: start, PCM frames, stop."""
    with TestClient(app).websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "start"}))
        for _ in range(3):
            ws.send_bytes(bytes(640))
        ws.send_text(json.dumps({"type": "stop"}))

        states, texts, audio_chunks, latency = [], {}, 0, None
        while True:
            message = ws.receive()
            if (raw := message.get("text")) is not None:
                payload = json.loads(raw)
                match payload["type"]:
                    case "state":
                        states.append(payload["state"])
                    case "transcript" | "agent_text":
                        texts[payload["type"]] = payload["text"]
                    case "latency":
                        latency = payload
            elif message.get("bytes") is not None:
                audio_chunks += 1
            if latency is not None and states[-1:] == ["LISTENING"]:
                break

    assert states == ["LISTENING", "TRANSCRIBING", "THINKING", "SPEAKING", "LISTENING"]
    assert texts["transcript"]
    assert texts["agent_text"]
    assert audio_chunks > 0
    assert latency["total_ms"] >= 0


def test_ws_rejects_a_malformed_message_without_dropping_the_call(fake_mode):
    with TestClient(app).websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "not-a-real-type"}))
        error = json.loads(ws.receive().get("text"))
        assert error["type"] == "error"

        # The connection must still work afterwards.
        ws.send_text(json.dumps({"type": "start"}))
        state = json.loads(ws.receive().get("text"))
        assert state == {"type": "state", "state": "LISTENING"}
