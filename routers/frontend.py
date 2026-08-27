# routers/frontend.py
"""
Serves the single-page frontend (index.html) at the site root.

This is intentionally the smallest router in the app — it exists purely so
app.py doesn't need to know about file I/O or HTML at all, matching the
same "app.py just wires routers together" philosophy as routers/pipeline.py.
"""
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["frontend"])


@router.get("/")
async def serve_ui():
    """
    Reads index.html fresh from disk on every request (not cached in memory) —
    means editing index.html and refreshing the browser shows changes
    immediately, no server restart needed. Fine for a dev/demo-scale app;
    would be worth caching if this ever needed to handle high request volume.
    """
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())