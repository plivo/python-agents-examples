"""
TTS Comparison Dashboard
=========================
FastAPI server serving a dashboard UI, replay page, and REST API for session data.

Run:
    python dashboard.py
    # Dashboard at http://localhost:8080
    # Replay at http://localhost:8080/replay/{session_id}
"""

import json
import os
from pathlib import Path

import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "sessions"
STATIC_DIR = BASE_DIR / "static"

router = APIRouter()

VOICE_LABELS = {
    "ara": "Ara",
    "eve": "Eve",
    "rex": "Rex",
    "sal": "Sal",
    "leo": "Leo",
}

PROVIDER_LABELS = {
    "xai": "xAI",
    "elevenlabs": "ElevenLabs",
    "cartesia": "Cartesia",
    "openai": "OpenAI",
}


def _read_session(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _session_summary(session: dict) -> dict:
    return {
        "session_id": session["session_id"],
        "mode": session["mode"],
        "started_at": session["started_at"],
        "ended_at": session.get("ended_at"),
        "config": session.get("config", {}),
        "summary": session.get("summary", {}),
        "recording_url": session.get("recording_url"),
    }


@router.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "dashboard.html"))


@router.get("/replay/{session_id}")
async def replay(session_id: str):
    # Verify session exists
    path = DATA_DIR / f"{session_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    return FileResponse(str(STATIC_DIR / "replay.html"))


@router.get("/api/sessions")
async def list_sessions():
    if not DATA_DIR.exists():
        return []
    sessions = []
    for p in sorted(DATA_DIR.glob("*.json")):
        s = _read_session(p)
        if s:
            sessions.append(_session_summary(s))
    return sessions


@router.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    path = DATA_DIR / f"{session_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    session = _read_session(path)
    if not session:
        raise HTTPException(status_code=500, detail="Failed to read session")
    return session


@router.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    path = DATA_DIR / f"{session_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    path.unlink()
    return {"message": "deleted"}


@router.get("/api/compare")
async def compare_sessions(ids: str):
    id_list = [i.strip() for i in ids.split(",") if i.strip()]
    if len(id_list) < 2 or len(id_list) > 5:
        raise HTTPException(status_code=400, detail="Provide 2-5 session IDs")
    results = []
    for sid in id_list:
        path = DATA_DIR / f"{sid}.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Session {sid} not found")
        s = _read_session(path)
        if not s:
            raise HTTPException(status_code=500, detail=f"Failed to read session {sid}")
        results.append(s)
    return results


@router.get("/api/providers/summary")
async def providers_summary():
    """Aggregate session data grouped by voice."""
    if not DATA_DIR.exists():
        return []
    groups: dict[str, list[dict]] = {}
    for p in DATA_DIR.glob("*.json"):
        s = _read_session(p)
        if not s:
            continue
        # Group by voice name (mode field stores voice)
        voice = s.get("mode", "unknown")
        groups.setdefault(voice, []).append(s)

    result = []
    for voice, sessions_list in sorted(groups.items()):
        n = len(sessions_list)
        summaries = [s.get("summary", {}) for s in sessions_list]
        configs = [s.get("config", {}) for s in sessions_list]
        scenario = next((c.get("scenario_label") for c in configs if c.get("scenario_label")), None)

        def avg_field(field):
            vals = [sm[field] for sm in summaries if sm.get(field) is not None]
            return round(sum(vals) / len(vals), 2) if vals else None

        entry = {
            "provider": voice,
            "label": VOICE_LABELS.get(voice, voice.title()),
            "scenario": scenario,
            "session_count": n,
            "avg_response_latency_ms": avg_field("avg_response_latency_ms"),
            "avg_stt_ttfb_ms": avg_field("avg_stt_ttfb_ms"),
            "avg_llm_ttfb_ms": avg_field("avg_llm_ttfb_ms"),
            "avg_tts_ttfb_ms": avg_field("avg_tts_ttfb_ms"),
            "avg_dead_air_s": avg_field("dead_air_s"),
        }
        result.append(entry)

    return result


def register_dashboard(target_app: FastAPI):
    target_app.include_router(router)
    target_app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


app = FastAPI(title="xAI Voice Showcase Dashboard")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
register_dashboard(app)


if __name__ == "__main__":
    port = int(os.getenv("DASHBOARD_PORT", "8080"))
    print(f"Dashboard: http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
