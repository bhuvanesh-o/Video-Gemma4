# app.py
import os
import uuid
import asyncio
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse

# Import your existing, unmodified pipeline modules
import storage
import progress
import video_preprocess
import step_1_segmentation
import step_2_video_slice_excel_timestamp
import step_3_yoloXs_images

app = FastAPI()

# 1. THE FRONTEND: Serve the HTML page when someone visits the site
@app.get("/")
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

# --- THE BACKGROUND WORKER ---
def run_pipeline(job_id: str):
    """Runs your AI models in the background so the website doesn't freeze."""
    try:
        progress.emit(f"🚀 Job {job_id} started!", job_id)
        
        # We pass the job_id into your steps just like you designed!
        progress.emit("--- Starting Step 1: Segmentation ---", job_id)
        step_1_segmentation.main(job_id=job_id)
        
        progress.emit("--- Starting Step 2: Slicing ---", job_id)
        step_2_video_slice_excel_timestamp.main(job_id=job_id)
        
        progress.emit("--- Starting Step 3: Tracking & OCR ---", job_id)
        step_3_yoloXs_images.main(job_id=job_id)

        progress.emit("✅ DONE", job_id)
    except Exception as e:
        progress.emit(f"❌ ERROR: {str(e)}", job_id)

# 2. THE UPLOAD ENDPOINT
@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Generate a random 8-character ID for this user's session
    job_id = str(uuid.uuid4())[:8] 
    

    # Save the file safely using your storage abstraction!
    saved_path = storage.save_upload(job_id, file.file, filename=file.filename)

    # ── VALIDATE + NORMALIZE BEFORE ANYTHING ELSE RUNS ────────────────────────
    # 1) Reject videos longer than 1 minute outright — no point burning compute
    #    on a job we're going to refuse.
    # 2) If resolution is above 720p, downscale it in place. Every downstream
    #    stage (step_1 plots, step_2 slices, step_3 YOLOX crops/annotated
    #    segments) reads from this same raw/ file, so this one pass is enough
    #    for the whole pipeline to operate in 720p. Videos already <= 720p are
    #    left untouched — storage isn't a concern there.
    try:
        video_preprocess.validate_and_prepare(saved_path, job_id=job_id)
    except ValueError as e:
        storage.delete_job(job_id)  # clean up the rejected upload, nothing to keep around
        return JSONResponse(status_code=400, content={"error": str(e)})
    

    # Tell FastAPI to run the pipeline in the background
    background_tasks.add_task(run_pipeline, job_id)
    
    # Instantly reply to the browser with the ID so it can start listening
    return {"job_id": job_id}

# app.py

# 3. THE LIVE PROGRESS ENDPOINT (Server-Sent Events)
@app.get("/stream/{job_id}")
async def stream_logs(job_id: str):
    async def event_generator():
        # FAST-CHECK: Did the browser try to reconnect to an ALREADY FINISHED job?
        full_history = progress.get_log(job_id)
        if any("✅ DONE" in log or "❌ ERROR" in log for log in full_history):
            # Send the final log immediately and shut down the generator
            yield "data: ✅ DONE\n\n"
            return

        # NORMAL OPERATION: The job is still running. Stream live updates.
        last_index = 0
        while True:
            logs = progress.get_log(job_id)
            if len(logs) > last_index:
                for log in logs[last_index:]:
                    yield f"data: {log}\n\n"
                    # If this specific line is the end flag, gracefully exit
                    if "✅ DONE" in log or "❌ ERROR" in log:
                        await asyncio.sleep(1) # Give UI a split-second to read it
                        return 
                last_index = len(logs)
            await asyncio.sleep(0.5) 
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

'''
        # Keep the connection quietly alive for 10 seconds 
        # This gives the UI browser plenty of time to call eventSource.close()
        # and prevents the browser from triggering an automatic reconnect.
        for _ in range(20):
            await asyncio.sleep(0.5)
            
'''


# 4. THE DOWNLOAD ENDPOINT
@app.get("/download/{job_id}")
async def download_results(job_id: str):
    # Locate the final excel sheet using your storage logic
    excel_path = storage.path_for(job_id, "final_assets", "Vehicle_Registry_Master.xlsx")
    if os.path.exists(excel_path):
        return FileResponse(excel_path, filename=f"Vehicle_Registry_{job_id}.xlsx")
    return {"error": "File not found or pipeline failed."}