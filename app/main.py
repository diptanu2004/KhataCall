"""FastAPI entry point. The /ws voice endpoint arrives in M1."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="KhataCall")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


# Mounted last so API routes take precedence over static files.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
