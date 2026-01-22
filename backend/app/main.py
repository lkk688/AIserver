from contextlib import asynccontextmanager
from fastapi import FastAPI
from backend.app.config.loader import load_config
from backend.app.dependencies import get_job_runner
import os

# Determine config path, default to relative path from where uvicorn is run (usually root)
config_path = os.getenv("APP_CONFIG_PATH", "backend/config.yaml")

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

@app.get("/health")
def health_check():
    return {"status": "ok", "config_loaded": config is not None}

from backend.app.api.routers import router as api_router
app.include_router(api_router, prefix="/api/v1")
