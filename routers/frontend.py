# routers/frontend.py
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["frontend"])


@router.get("/")
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())