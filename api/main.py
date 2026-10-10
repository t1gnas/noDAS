import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api.context import MODEL_NAME, build_context
from api.session import SPEEDS, run_live
from das.config import model_path
from das.export import RESULTS_FILE


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.ctx = await asyncio.to_thread(build_context)
    yield


app = FastAPI(title="noDAS demo API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "model": MODEL_NAME}


@app.get("/api/scenarios")
def scenarios() -> dict:
    return {"scenarios": app.state.ctx["cards"], "classes": app.state.ctx["classes"], "speeds": list(SPEEDS)}


@app.get("/api/results")
def results() -> dict:
    path = model_path(app.state.ctx["cfg"], RESULTS_FILE)
    if not path.exists():
        raise HTTPException(404, "results.json missing; run `docker compose run --rm das export`")
    return json.loads(path.read_text())


@app.websocket("/api/live/{scenario_id}")
async def live(ws: WebSocket, scenario_id: str, speed: float = 5.0) -> None:
    await ws.accept()
    try:
        await run_live(ws, app.state.ctx, scenario_id, speed)
    except AssertionError as e:
        await ws.send_json({"type": "error", "message": str(e)})
    finally:
        with suppress(RuntimeError, WebSocketDisconnect):
            await ws.close()
