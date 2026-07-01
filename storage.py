# storage.py
"""
Local-disk storage abstraction for the traffic pipeline web app.

Every pipeline stage and the FastAPI layer should go through THIS module
instead of touching open()/shutil directly for job files. That way, if R2
(or any other remote storage) gets added later, only the internals of these
functions change — nothing that calls them needs to be rewritten.

Folder layout per job:
storage/jobs/{job_id}/
    raw/                        <- uploaded source video
    segments/{video_base}/segment_N.mp4
    plots/                      <- step 1 PNGs
    assets/truck_crops/
    assets/plate_crops/
    assets/annotated_segments/
    segment_timestamps.xlsx
    Vehicle_Registry_Master.xlsx
"""

import os
import shutil

# Always anchored to wherever THIS file physically lives, regardless of
# what folder VS Code/the terminal happens to be "in" when you run things.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_ROOT = os.path.join(BASE_DIR, "storage", "jobs")


def job_dir(job_id):
    """Root folder for a given job. Created if it doesn't exist."""
    path = os.path.join(STORAGE_ROOT, job_id)
    os.makedirs(path, exist_ok=True)
    return path


def path_for(job_id, *relative_parts):
    """
    Build (and ensure the parent folder exists for) a path inside a job's
    folder. Use this everywhere instead of manually joining paths, e.g.:
        path_for(job_id, "segments", "video1", "segment_1.mp4")
        path_for(job_id, "raw", "input.mp4")
    """
    full_path = os.path.join(job_dir(job_id), *relative_parts)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    return full_path


def save_upload(job_id, file_obj, filename="input.mp4"):
    """
    Stream an uploaded file (e.g. FastAPI's UploadFile.file) to disk under
    this job's raw/ folder without loading the whole thing into memory.
    Returns the saved file's local path.
    """
    dest_path = path_for(job_id, "raw", filename)
    with open(dest_path, "wb") as out:
        shutil.copyfileobj(file_obj, out)
    return dest_path


def list_files(job_id, *relative_parts):
    """List files inside a subfolder of a job's directory."""
    folder = path_for(job_id, *relative_parts)
    if not os.path.isdir(folder):
        return []
    return sorted(os.listdir(folder))


def delete_job(job_id):
    """
    Wipe an entire job's folder. Call this once results have been
    downloaded / after some retention window, so disk doesn't fill up —
    this is your local-disk equivalent of the R2 'recycle storage' idea.
    """
    path = os.path.join(STORAGE_ROOT, job_id)
    if os.path.exists(path):
        shutil.rmtree(path)


def delete_file(job_id, *relative_parts):
    """Delete a single file inside a job's folder, if it exists."""
    path = path_for(job_id, *relative_parts)
    if os.path.exists(path):
        os.remove(path)