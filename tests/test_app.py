from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app


def test_healthz_reports_ok():
    assert TestClient(app).get("/healthz").json() == {"status": "ok"}


def test_index_page_is_served():
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "KhataCall" in response.text


def test_settings_defaults_match_documented_gnani_values():
    settings = Settings(_env_file=None)
    assert settings.gnani_base_url == "https://api.vachana.ai"
    assert settings.stt_language == "en-IN"
    assert settings.use_fakes is False
