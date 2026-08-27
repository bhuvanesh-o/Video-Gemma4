# app.py
"""
Top-level FastAPI application entrypoint. Wires together the two routers
(frontend.py for serving the UI, pipeline.py for the actual video-processing
API) and runs a background cleanup loop for old job folders.

Run with: uvicorn app:app --reload
"""
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from routers import pipeline, frontend
import storage

import os
from dotenv import load_dotenv

# Load environment variables (OPENROUTER_API_KEY, HF_TOKEN, etc.) from a
# local .env file at startup. This means those keys no longer need to be
# set manually in every terminal session before running uvicorn — .env
# should be in .gitignore and never committed, since it holds real secrets.
load_dotenv()


CLEANUP_INTERVAL_SECONDS = 3600  # check hourly, delete anything older than 24h

async def _cleanup_loop():
    """
    Background task that runs for the entire lifetime of the server process.
    Every hour, sweeps storage/jobs/ and deletes any job folder whose last
    modification time is older than 24h — see storage.cleanup_old_jobs()
    for the actual deletion logic. Wrapped in try/except so a single sweep
    failing (e.g. a file locked by another process) doesn't kill the loop
    permanently — it just logs and tries again next hour.
    """
    while True:
        try:
            storage.cleanup_old_jobs(max_age_hours=24)
        except Exception as e:
            print(f"[cleanup] error: {e}")
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI's modern startup/shutdown hook. On startup, launches the cleanup
    loop as a background asyncio task (doesn't block the server from
    accepting requests). On shutdown, cancels that task cleanly so the
    process can exit without hanging.
    """
    task = asyncio.create_task(_cleanup_loop())
    yield
    task.cancel()

app = FastAPI(lifespan=lifespan)
app.include_router(frontend.router)   # serves "/" -> index.html
app.include_router(pipeline.router)   # serves everything under "/api/..."