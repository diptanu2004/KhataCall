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
        ws.receive()  # the opening ledger snapshot
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

    # The canned turn is "sharma ji ka dedh sau udhaar likho", a write, so the
    # session passes through CONFIRMING and reads the amount back rather than
    # writing anything (CLAUDE.md 5).
    assert states == ["LISTENING", "TRANSCRIBING", "THINKING", "CONFIRMING", "SPEAKING", "LISTENING"]
    assert texts["transcript"]
    assert "150 rupaye" in texts["agent_text"]
    assert texts["agent_text"].endswith("?"), "a write must be read back as a question"
    assert audio_chunks > 0
    assert latency["total_ms"] >= 0


def test_ws_rejects_a_malformed_message_without_dropping_the_call(fake_mode):
    with TestClient(app).websocket_connect("/ws") as ws:
        opening = json.loads(ws.receive().get("text"))
        assert opening["type"] == "ledger_update", "the book is pushed on connect"

        ws.send_text(json.dumps({"type": "not-a-real-type"}))
        error = json.loads(ws.receive().get("text"))
        assert error["type"] == "error"

        # The connection must still work afterwards.
        ws.send_text(json.dumps({"type": "start"}))
        state = json.loads(ws.receive().get("text"))
        assert state == {"type": "state", "state": "LISTENING"}


def test_ws_confirmation_commits_and_pushes_a_ledger_update(fake_mode, tmp_path, monkeypatch):
    """The whole M2 loop over the real endpoint: propose, read back, "haan", write."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    get_settings.cache_clear()

    def drain(ws, until: str) -> list[dict]:
        seen = []
        while True:
            message = ws.receive()
            if (raw := message.get("text")) is None:
                continue
            payload = json.loads(raw)
            seen.append(payload)
            if payload["type"] == until:
                return seen

    with TestClient(app).websocket_connect("/ws") as ws:
        opening = json.loads(ws.receive().get("text"))
        assert opening["type"] == "ledger_update"

        # Turn one: the proposal and its read-back.
        ws.send_text(json.dumps({"type": "start"}))
        ws.send_bytes(bytes(640))
        ws.send_text(json.dumps({"type": "stop"}))
        first = drain(ws, "latency")
        assert any(m["type"] == "state" and m["state"] == "CONFIRMING" for m in first)
        assert not [m for m in first if m["type"] == "ledger_update"], "nothing written before yes"

        # Turn two: "haan".
        ws.send_text(json.dumps({"type": "start"}))
        ws.send_bytes(bytes(640))
        ws.send_text(json.dumps({"type": "stop"}))
        second = drain(ws, "latency")

    updates = [m for m in second if m["type"] == "ledger_update"]
    assert updates, "a committed write must push the ledger to the UI"
    sharma = next(c for c in updates[0]["customers"] if c["name"] == "Sharma ji")
    assert sharma["balance_paise"] == 15_000

    agent_text = next(m["text"] for m in second if m["type"] == "agent_text")
    assert "Likh diya" in agent_text
