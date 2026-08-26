# app.py
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from routers import pipeline, frontend
import storage


import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()



CLEANUP_INTERVAL_SECONDS = 3600  # check hourly, delete anything older than 24h

async def _cleanup_loop():
    while True:
        try:
            storage.cleanup_old_jobs(max_age_hours=24)
        except Exception as e:
            print(f"[cleanup] error: {e}")
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_cleanup_loop())
    yield
    task.cancel()

app = FastAPI(lifespan=lifespan)
app.include_router(frontend.router)
app.include_router(pipeline.router)