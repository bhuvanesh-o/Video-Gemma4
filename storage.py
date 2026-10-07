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

import time

import re

# Always anchored to wherever THIS file physically lives, regardless of
# what folder VS Code/the terminal happens to be "in" when you run things.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_ROOT = os.path.join(BASE_DIR, "storage", "jobs")


_JOB_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9_-]{1,64}$"
)


def _validate_job_id(job_id):
    if not _JOB_ID_PATTERN.fullmatch(job_id):
        raise ValueError(
            "Invalid job ID."
        )


def job_dir(job_id):
    """Root folder for a given job. Created if it doesn't exist."""
    path = os.path.join(STORAGE_ROOT, job_id)
    os.makedirs(path, exist_ok=True)
    return path


def _safe_join(base, *parts):
    base = os.path.abspath(base)
    full = os.path.abspath(
        os.path.join(
            base,
            *parts,
        )
    )

    if os.path.commonpath(
        [base, full]
    ) != base:
        raise ValueError(
            "Invalid storage path."
        )

    return full


def path_for(job_id, *relative_parts):
    """
    Build (and ensure the parent folder exists for) a path inside a job's
    folder. Use this everywhere instead of manually joining paths, e.g.:
        path_for(job_id, "segments", "video1", "segment_1.mp4")
        path_for(job_id, "raw", "input.mp4")
    """

    '''
    It only creates the parent folder of whatever path you build. That works fine for something 
    like path_for(job_id, "raw", "input.mp4") — the parent (raw/) gets created, and input.mp4 itself 
    is a file you're about to write, so that's correct.

    But SAVE_DIR = path_for(job_id, "plots") is being used as a folder itself, 
    not a file inside a folder. So os.path.dirname(full_path) here computes the dirname of 
    .../local_test/plots, which is .../local_test — it creates the job folder, but never actually 
    creates the plots folder itself. Then when render_segment_plot tries to save a PNG inside plots/, 
    that folder doesn't exist yet → crash.

    '''
    full_path = _safe_join(job_dir(job_id),*relative_parts,)    
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    return full_path





def dir_for(job_id, *relative_parts):
    """
    Like path_for(), but for paths that are themselves folders (not files).
    Creates the folder itself, not just its parent.
    """
    full_path = os.path.join(job_dir(job_id), *relative_parts)
    os.makedirs(full_path, exist_ok=True)
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


def cleanup_old_jobs(max_age_hours=24):
    """
    Deletes any job folder under storage/jobs/ whose last-modified time is
    older than max_age_hours. Client demo links get shared and revisited
    unpredictably, so old uploads (someone else's footage, crops, excel
    files) need to age out on their own instead of piling up forever.
    """
    if not os.path.exists(STORAGE_ROOT):
        return
    cutoff = time.time() - (max_age_hours * 3600)
    for job_id in os.listdir(STORAGE_ROOT):
        job_path = os.path.join(STORAGE_ROOT, job_id)
        if not os.path.isdir(job_path):
            continue
        try:
            if os.path.getmtime(job_path) < cutoff:
                shutil.rmtree(job_path)
        except OSError:
            continue  # folder mid-write or already gone — skip rather than crash the whole sweep