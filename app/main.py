"""FastAPI entry point: static UI plus the /ws voice endpoint."""

import json
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from app.adapters.factory import build_adapters
from app.config import get_settings
from app.ledger.db import connect
from app.ledger.repo import Ledger
from app.pipeline.session import Session
from app.protocol import CLIENT_SAMPLE_RATE, ErrorMessage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="KhataCall")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "mode": _mode(settings),
        "client_sample_rate": str(CLIENT_SAMPLE_RATE),
    }


def _mode(settings) -> str:
    if settings.use_fakes:
        return "fakes"
    return "gnani (canned Evon)" if settings.fake_llm else "gnani"


@app.websocket("/ws")
async def voice(websocket: WebSocket) -> None:
    await websocket.accept()
    settings = get_settings()
    stt, llm, tts = build_adapters(settings)
    # One connection per call. SQLite connections are not safe to share across
    # threads, and a call is short-lived, so this is both simpler and correct.
    database = connect(settings.db_path)
    session_id = uuid.uuid4().hex[:8]
    session = Session(
        stt=stt,
        llm=llm,
        tts=tts,
        send_json=websocket.send_json,
        send_bytes=websocket.send_bytes,
        stt_language=settings.stt_language,
        tts_language=settings.tts_language,
        tts_voice=settings.tts_voice,
        client_sample_rate=CLIENT_SAMPLE_RATE,
        ledger=Ledger(database),
        session_id=session_id,
    )
    log.info("session %s open (mode=%s, db=%s)", session_id, _mode(settings), settings.db_path)
    await session.send_opening_ledger()

    try:
        while True:
            packet = await websocket.receive()
            if packet["type"] == "websocket.disconnect":
                break
            if (pcm := packet.get("bytes")) is not None:
                await session.handle_audio(pcm)
            elif (text := packet.get("text")) is not None:
                try:
                    await session.handle_json(json.loads(text))
                except Exception as e:
                    # A bad message should not kill the call.
                    log.exception("message failed: %s", text[:200])
                    await websocket.send_json(ErrorMessage(message=str(e)).model_dump(mode="json"))
    except WebSocketDisconnect:
        pass
    finally:
        await session.aclose()
        database.close()
        log.info("session %s closed", session_id)


# Mounted last so API routes take precedence over static files.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
