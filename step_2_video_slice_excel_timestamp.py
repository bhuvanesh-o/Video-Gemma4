# -*- coding: utf-8 -*-
"""
step_2_video_slice_excel_timestamp.py

This is where we SLICE THE MAIN VIDEO INTO SEGMENTS based on the calculations
it is again doing (yes there are redundancies of the same process... like
segmentation of timestamps and so on... which I will change and optimise
later) and save the segments in a folder.

REFACTOR NOTE: The long processing chunks (per-video Cosine extraction,
peak/segment detection, slicing+Excel logging, Excel merge) now live in
step_2_functions_video_slice_excel_timestamp.py as callable functions. This file just holds your
config/tuning parameters and orchestrates the single processing loop by
calling into that library.

The whole thing is wrapped in main() so this file can either be run directly
(`python step_2_video_slice_excel_timestamp.py`) or imported and called from a
higher-level orchestrator script, e.g.:
    import step_2_video_slice_excel_timestamp
    step_2_video_slice_excel_timestamp.main(job_id="local_test")

REFACTOR NOTE 2: Hardcoded D:\\Traffic_Control paths are replaced by storage.path_for()
so every run is isolated to its own job folder. print() is replaced by emit() from
progress.py so progress can be streamed to a browser later via FastAPI SSE while
still printing locally when run from the command line.

NOTE: Before running, install dependencies in your terminal:
    pip install opencv-python numpy scipy pandas openpyxl

Make sure step_2_functions_video_slice_excel_timestamp.py, storage.py, and progress.py
are in the same folder as this script (or somewhere on your PYTHONPATH) so the
imports below resolve.
"""
import os
import pandas as pd

from step_2_functions_video_slice_excel_timestamp import (
    extract_cosine_curve,
    find_cosine_segments,
    slice_video_segments,
    save_segment_excel,
)
from storage import path_for, dir_for
from progress import emit


def main(job_id):
    # ── JOB-SCOPED PATH CONFIG ────────────────────────────────────────────────
    # Replaces hardcoded D:\Traffic_Control paths — same raw/ folder step 1
    # reads from, so this stage picks up right where step 1 left off.
    MAIN_VIDEO_DIR = path_for(job_id, "raw")
    SEGMENT_DIR = dir_for(job_id, "trial_video_segments")
    EXCEL_PATH = path_for(job_id, "segment_timestamps.xlsx")

    if not os.path.exists(MAIN_VIDEO_DIR):
        raise FileNotFoundError(f"⚠️ Could not find processing folder: {MAIN_VIDEO_DIR}.")

    all_files = os.listdir(MAIN_VIDEO_DIR)
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_paths = [f for f in all_files if f.lower().endswith(video_extensions)]
    emit(f"📦 Found {len(video_paths)} videos. Starting Engine...", job_id=job_id)

    # 2. HYPERPARAMETERS
    BINS = 64
    SAVGOL_WINDOW = 7
    SAVGOL_POLY = 3
    SAFE_NOISE_FLOOR = 0.0005
    PEAK_PROMINENCE_FACTOR = 0.12
    MIN_PEAK_HEIGHT = 0.15
    MIN_PEAK_DISTANCE_SECS = 5
    BASE_SLOPE_CUTOFF = 0.20

    # 🌟 CRITICAL: Memory list to store our Excel data
    excel_metadata_list = []

    # ==============================================================================
    # ── STEP 4: MASS PROCESSING LOOP ──────────────────────────────────────────────
    # ==============================================================================
    for idx, v_name in enumerate(video_paths, 1):
        v_path = os.path.join(MAIN_VIDEO_DIR, v_name)
        emit(f"\n🎬 [Asset {idx}/{len(video_paths)}] Processing: {v_name}", job_id=job_id)

        result = extract_cosine_curve(v_path, savgol_window=SAVGOL_WINDOW, savgol_poly=SAVGOL_POLY, bins=BINS, job_id=job_id)
        if result is None:
            continue  # Corrupt/unopenable video, or too few frames to compare

        timestamps, fps, cosine_smoothed = result
        segments = find_cosine_segments(
            cosine_smoothed,
            noise_floor=SAFE_NOISE_FLOOR,
            peak_prominence_factor=PEAK_PROMINENCE_FACTOR,
            min_peak_height=MIN_PEAK_HEIGHT,
            min_peak_distance_secs=MIN_PEAK_DISTANCE_SECS,
            base_slope_cutoff=BASE_SLOPE_CUTOFF
        )

        if not segments:
            emit("  📍 No significant transitions found. Skipping.", job_id=job_id)
            continue

        excel_rows = slice_video_segments(v_path, v_name, segments, timestamps, fps, SEGMENT_DIR, job_id=job_id)
        excel_metadata_list.extend(excel_rows)

    # ==============================================================================
    # ── STEP 5: EXCEL SPREADSHEET CREATION (THE MAP FOR YOLOX) ────────────────────
    # ==============================================================================
    save_segment_excel(excel_metadata_list, EXCEL_PATH, job_id=job_id)
    emit("🎉 PIPELINE RUN COMPLETE!", job_id=job_id)


if __name__ == "__main__":
    # Local manual test: run step_1_segmentation.py first (or drop a video into
    # storage/jobs/local_test/raw/ manually), then run this file directly.
    main(job_id="local_test")