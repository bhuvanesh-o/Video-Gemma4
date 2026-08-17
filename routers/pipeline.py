# routers/pipeline.py
import os
import json
import uuid
import asyncio
from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
import pandas as pd

import zipfile

# Import your existing, unmodified pipeline modules
import storage
import progress
import video_preprocess
import step_1_segmentation
import step_2_video_slice_excel_timestamp
import step_3_yoloXs_images
import step_4_prompt_document


# REFACTOR NOTE (APIRouter split): this used to live directly on the FastAPI()
# app in app.py. Moved here so app.py only wires routers together instead of
# holding all route logic itself. prefix="/api" means every route below is
# actually reachable at /api/upload, /api/stream/{job_id}, /api/download/{job_id}
# — not at the bare paths shown in each @router decorator.
router = APIRouter(prefix="/api", tags=["pipeline"])


# --- THE BACKGROUND WORKER ---
def run_pipeline(job_id: str):
    """
    Runs your AI models in the background so the website doesn't freeze.
    REFACTOR NOTE: emit_stage_complete() is called after each step finishes,
    so the frontend's stage trail (segmentation -> slicing -> detection) can
    mark a stage "done" and move the spotlight to the next one, instead of
    guessing based on log text.
    """
    try:
        step_1_segmentation.main(job_id=job_id)
        progress.emit_stage_complete("segmentation", job_id)

        step_2_video_slice_excel_timestamp.main(job_id=job_id)
        progress.emit_stage_complete("slicing", job_id)

        step_3_yoloXs_images.main(job_id=job_id)
        progress.emit_stage_complete("detection", job_id)

        step_4_prompt_document.main(job_id=job_id)          
        progress.emit_stage_complete("describing", job_id)   

        progress.emit_done(job_id)
        
    except Exception as e:
        progress.emit_error(str(e), job_id)

# 2. THE UPLOAD ENDPOINT
@router.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Generate a random 8-character ID for this user's session
    job_id = str(uuid.uuid4())[:8] 
    
    # Save the file safely using your storage abstraction!
    saved_path = storage.save_upload(job_id, file.file, filename=file.filename)

    # ── VALIDATE + NORMALIZE BEFORE ANYTHING ELSE RUNS ────────────────────────
    # 1) Reject videos longer than 1 minute or shorter than 720p outright — no
    #    point burning compute on a job we're going to refuse.
    # 2) If resolution is above 720p, downscale it in place. Every downstream
    #    stage (step_1 plots, step_2 slices, step_3 YOLOX crops/annotated
    #    segments) reads from this same raw/ file, so this one pass is enough
    #    for the whole pipeline to operate in 720p.
    try:
        video_preprocess.validate_and_prepare(saved_path, job_id=job_id)
    except ValueError as e:
        storage.delete_job(job_id)  # clean up the rejected upload, nothing to keep around
        return JSONResponse(status_code=400, content={"error": str(e)})

    # Tell FastAPI to run the pipeline in the background
    background_tasks.add_task(run_pipeline, job_id)
    
    # Instantly reply to the browser with the ID so it can start listening
    return {"job_id": job_id}

# 3. THE LIVE PROGRESS ENDPOINT (Server-Sent Events)
# REFACTOR NOTE (structured events): now streams JSON event objects
# ({"type": ..., "stage": ..., "text": ...}) instead of raw log strings, so
# the frontend can drive a proper stage-by-stage narrative UI instead of a
# scrolling text log. Completion is detected via event["type"] rather than
# string-matching "DONE"/"ERROR" inside message text.
@router.get("/stream/{job_id}")
async def stream_logs(job_id: str):
    async def event_generator():
        # FAST-CHECK: Did the browser try to reconnect to an ALREADY FINISHED job?
        full_history = progress.get_log(job_id)
        if any(e["type"] in ("done", "error") for e in full_history):
            # Replay the full history so the UI can rebuild its final state,
            # then close — this job has nothing left to stream.
            for e in full_history:
                yield f"data: {json.dumps(e)}\n\n"
            return

        # NORMAL OPERATION: The job is still running. Stream live updates.
        last_index = 0
        while True:
            logs = progress.get_log(job_id)
            if len(logs) > last_index:
                for e in logs[last_index:]:
                    yield f"data: {json.dumps(e)}\n\n"
                    # If this event is the end flag, gracefully exit
                    if e["type"] in ("done", "error"):
                        await asyncio.sleep(1)  # Give UI a split-second to read it
                        return
                last_index = len(logs)
            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# 4. THE DOWNLOAD ENDPOINT
@router.get("/download/{job_id}")
async def download_results(job_id: str):
    # Locate the final excel sheet using your storage logic
    excel_path = storage.path_for(job_id, "final_assets", "Vehicle_Registry_Master.xlsx")
    if os.path.exists(excel_path):
        return FileResponse(excel_path, filename=f"Vehicle_Registry_{job_id}.xlsx")
    return {"error": "File not found or pipeline failed."}


import re

def _extract_image_url(cell_value, job_id, crop_type):
    """
    Converts an Excel hyperlink formula into a browser-usable URL.

    The Excel cell looks like: =HYPERLINK("C:\...\storage\jobs\{job_id}\final_assets\truck_crops\TRUCK_1_1_truck.jpg", "View Truck")
    That local filesystem path means nothing to a browser running on a
    different machine (or even the same machine, since browsers can't just
    open arbitrary disk paths for security reasons). So instead of sending
    that raw path to the frontend, we:
      1. Pull just the filename back out of the formula string via regex
      2. Rebuild it as a URL pointing at our own /api/image/... route,
         which knows how to actually locate + serve that file from disk

    crop_type is passed in explicitly (rather than guessed from the cell)
    because we already know which column we're processing when we call
    this — "Plate File" always maps to plate_crops/, "Truck File" always
    maps to truck_crops/. Keeps the whitelist check on the serving route
    simple and correct.

    Returns None if the cell isn't a real hyperlink (e.g. it's the string
    "NO_IMAGE" / "NO_PLATE", meaning no crop was ever saved for that row) —
    the frontend treats None as "show a placeholder dash instead of a thumbnail".
    """
    if not isinstance(cell_value, str) or not cell_value.startswith("=HYPERLINK"):
        return None

    # The formula wraps both the path and the display text in double quotes:
    #   =HYPERLINK("<path>", "<label>")
    # This regex grabs the FIRST quoted string, which is always the path.
    match = re.search(r'"([^"]+)"', cell_value)
    if not match:
        return None

    # os.path.basename strips off the full local directory structure and
    # keeps just the actual filename (e.g. "TRUCK_1_1_truck.jpg") — we don't
    # want to leak the server's local folder layout to the client, and the
    # /image route only needs the filename since it already knows job_id
    # and crop_type from the URL path.
    filename = os.path.basename(match.group(1))
    return f"/api/image/{job_id}/{crop_type}/{filename}"


#5
@router.get("/preview/{job_id}")
async def preview_results(job_id: str):
    """
    Returns the finished job's Excel registry as JSON so the frontend can
    render it as an in-page table, instead of forcing the user to download
    the .xlsx just to see what's in it.
    """
    excel_path = storage.path_for(job_id, "final_assets", "Vehicle_Registry_Master.xlsx")
    if not os.path.exists(excel_path):
        # Shouldn't normally happen — save_vehicle_registry always writes
        # a file (even an empty headers-only one) once a job reaches "done".
        # This guards against someone hitting /preview for a bad/unknown job_id.
        return JSONResponse(status_code=404, content={"error": "Results not found."})

    # data_only=False forces openpyxl to return the FORMULA TEXT
    # (=HYPERLINK("...", "...")) instead of the cached calculated result —
    # which doesn't exist here since this file was written by openpyxl,
    # never opened/recalculated by real Excel.
    df = pd.read_excel(excel_path, engine="openpyxl", engine_kwargs={"data_only": False})
    df = df.astype(object).where(df.notna(), None)

    rows = []
    for _, row in df.iterrows():
        # .to_dict() keys each value by its COLUMN NAME rather than position —
        # this is what lets the frontend look up row["Plate File"] reliably,
        # instead of having to remember "plate file is always the 6th item".
        row_dict = row.to_dict()

        # Overwrite the raw Excel formula string with a real, clickable URL
        # (or None, if there was never a crop saved for this row).
        row_dict["Plate File"] = _extract_image_url(row_dict.get("Plate File"), job_id, "plate_crops")
        row_dict["Truck File"] = _extract_image_url(row_dict.get("Truck File"), job_id, "truck_crops")

        rows.append(row_dict)

    return {"columns": list(df.columns), "rows": rows}


#6
@router.get("/image/{job_id}/{crop_type}/{filename}")
async def get_crop_image(job_id: str, crop_type: str, filename: str):
    """
    Serves a single truck/plate crop image file from a job's final_assets/
    folder over HTTP, so <img src="..."> tags in the browser can actually
    load it.

    crop_type is restricted to a hardcoded whitelist (not just any string
    the client sends) — this is a basic path-traversal guard. Without this
    check, a malicious request like crop_type="../../../../etc" could try
    to walk storage.path_for() outside the intended folder. Locking it to
    exactly these two known subfolder names closes that off.
    """
    if crop_type not in ("truck_crops", "plate_crops"):
        return JSONResponse(status_code=400, content={"error": "Invalid crop type."})

    image_path = storage.path_for(job_id, "final_assets", crop_type, filename)
    if not os.path.exists(image_path):
        return JSONResponse(status_code=404, content={"error": "Image not found."})

    return FileResponse(image_path)


# -----------------------------------------------------------------------------
# NEW ROUTES: SERVING YOLOX ANNOTATED VIDEOS
# -----------------------------------------------------------------------------

@router.get("/videos/{job_id}")
async def list_annotated_videos(job_id: str):
    """
    Returns a list of URLs for all annotated video segments generated by Step 3.
    The frontend calls this when the pipeline finishes to know what videos to display.
    """
    # 1. Look inside the 'annotated_segments' folder for this specific job
    files = storage.list_files(job_id, "final_assets", "annotated_segments")
    
    # 2. Filter for only .mp4 files and create a URL path for each one
    urls = [f"/api/video/{job_id}/{f}" for f in files if f.endswith(".mp4")]
    
    # 3. Send the list of URLs back to the frontend as JSON
    return {"videos": urls}

@router.get("/video/{job_id}/{filename}")
async def serve_annotated_video(job_id: str, filename: str):
    """
    Serves the actual physical .mp4 file bytes so the browser's <video> tag can play it.
    """
    # 1. Build the exact local file path on the server
    video_path = storage.path_for(job_id, "final_assets", "annotated_segments", filename)
    
    # 2. Safety check: if the file doesn't exist, return a 404 Not Found
    if not os.path.exists(video_path):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"error": "Video not found."})
        
    # 3. Return a FileResponse, which FastAPI automatically handles as a streamable video file
    from fastapi.responses import FileResponse
    return FileResponse(video_path, media_type="video/mp4")



@router.get("/download-videos/{job_id}")
async def download_annotated_videos(job_id: str):
    """
    Zips every annotated segment video for this job into one file and
    serves that — avoids ever needing the browser to decode/play these
    inline, so codec compatibility (mp4v vs H.264) stops being a concern.
    """
    video_dir = storage.path_for(job_id, "final_assets", "annotated_segments")
    if not os.path.exists(video_dir):
        return JSONResponse(status_code=404, content={"error": "No annotated videos found."})

    files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    if not files:
        return JSONResponse(status_code=404, content={"error": "No annotated videos found."})

    zip_path = storage.path_for(job_id, "final_assets", "annotated_segments.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in files:
            zf.write(os.path.join(video_dir, f), arcname=f)

    return FileResponse(zip_path, filename=f"Annotated_Videos_{job_id}.zip")