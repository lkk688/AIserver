import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.config.loader import load_config
from backend.app.dependencies import get_job_runner
import os

# Determine config path, default to relative path from where uvicorn is run (usually root)
config_path = os.getenv("APP_CONFIG_PATH", str(_ROOT / "backend" / "config.yaml"))

try:
    config = load_config(config_path)
except Exception as e:
    print(f"Failed to load config from {config_path}: {e}")
    # In a real app, we might want to exit here, but for now we'll let it run so tests can inspect it
    config = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    runner = get_job_runner()
    runner.start()
    yield
    # Shutdown
    runner.stop()

app = FastAPI(title="Local Search RAG", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "config_loaded": config is not None}

from backend.app.api.routers import router as api_router
from backend.app.api.llm_router import router as llm_router
from BatchAgent.agent_route_fastapi import router as agent_router

app.include_router(api_router, prefix="/api/v1")
app.include_router(llm_router, prefix="/api/v1")
app.include_router(agent_router, prefix="/api/v1")

