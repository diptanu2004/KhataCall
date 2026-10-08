"""Settings loaded from the environment / .env. See .env.example for every key."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Gnani API: Prisma (STT) + Timbre (TTS). Base URL and auth header per https://docs.gnani.ai
    gnani_api_key: SecretStr = SecretStr("")
    gnani_base_url: str = "https://api.vachana.ai"

    # Prisma: code-mixed English-Hindi is supported via en-IN (DECISIONS.md D-002).
    stt_language: str = "en-IN"

    # Timbre. Poorvi is the only hi-en (Hinglish) voice in the catalog. VERIFY: hi-en quality vs hi-IN.
    tts_voice: str = "Poorvi"
    tts_language: str = "hi-en"
    tts_sample_rate: int = 16_000

    # Evon v3.3 over an OpenAI-compatible /v1/chat/completions API (DECISIONS.md D-001).
    evon_base_url: str = "http://localhost:8001/v1"
    evon_api_key: SecretStr = SecretStr("")
    # VERIFY: model id taken from a participant repo, not from Gnani; confirm on Hugging Face.
    evon_model: str = "gnani/gnani-evon-v3.3-30B-A3B"

    db_path: str = "khatacall.db"
    # All three adapters faked: no network, no credits.
    use_fakes: bool = False
    # Only Evon faked, so real Prisma and Timbre can be demoed before the weights land.
    fake_llm: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
