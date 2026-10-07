# -*- coding: utf-8 -*-

"""
Convenience launcher for the FastAPI web application.

Recommended:
    python -m uvicorn app:app --reload

Alternative:
    python main.py

The actual video-processing pipeline is started through
routers/pipeline.py after a video is uploaded through the web interface.
"""


import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
