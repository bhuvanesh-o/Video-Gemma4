# -*- coding: utf-8 -*-
"""
step_1_segmentation.py

We are looking through the MAIN VIDEO and plotting the CHANGE POINT GRAPH and
we are updating all the SEGMENT TIMES INTO EXCEL SHEET and we will also print
the timings of each segment in the O/P also.

REFACTOR NOTE: The long processing chunks (per-video extraction, peak/segment
detection, plotting, Excel merge) now live in step_1_functions_segmentation.py as
callable functions. This file just holds your config/tuning parameters and
orchestrates the two main loops by calling into that library.

The whole thing is wrapped in main() so this file can either be run directly
(`python step_1_segmentation.py`) or imported and called from a higher-level
orchestrator script, e.g.:
    import step_1_segmentation
    step_1_segmentation.main(job_id="local_test")

REFACTOR NOTE 2: Hardcoded D:\\Traffic_Control paths are replaced by storage.path_for()
so every run is isolated to its own job folder. print() is replaced by emit() from
progress.py so progress can be streamed to a browser later via FastAPI SSE while
still printing locally when run from the command line.

REFACTOR NOTE 3 (structured events): every emit() call below now also passes
stage="segmentation" and a short ui_message for the frontend's narrative UI.

NOTE: Before running, install dependencies in your terminal:
    pip install opencv-python numpy matplotlib scipy pandas openpyxl

Make sure step_1_functions_segmentation.py, storage.py, and progress.py are in the
same folder as this script (or somewhere on your PYTHONPATH) so the imports below resolve.
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg') # Forces matplotlib to run in pure file-saving headless mode

from step_1_functions_segmentation import (
    extract_video_metrics,
    scale_and_segment,
    log_segments,
    render_segment_plot,
    save_segment_excel,
)
from storage import path_for, dir_for
from progress import emit


def main(job_id):
    # ── JOB-SCOPED PATH CONFIG ────────────────────────────────────────────────
    # Replaces hardcoded D:\Traffic_Control paths with job-scoped folders so
    # every upload/run is fully isolated from every other job on the server.
    # path_for() auto-creates the folder if it doesn't exist yet.
    VIDEO_DIR = dir_for(job_id, "raw")       # uploaded video(s) live here
    SAVE_DIR  = dir_for(job_id, "plots")     # segmentation PNGs go here
    EXCEL_PATH = path_for(job_id, "segment_timestamps.xlsx")

    if not os.path.exists(VIDEO_DIR):
        raise FileNotFoundError(f"⚠️ Could not find folder: {VIDEO_DIR}. Was a video uploaded for this job?")

    all_files = os.listdir(VIDEO_DIR)
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_paths = [f for f in all_files if f.lower().endswith(video_extensions)]

    emit(f"📦 Found {len(video_paths)} videos. Ingesting mass-processing engine...", job_id=job_id,
         stage="segmentation", ui_message="Scanning your footage for scene changes...")

    BINS = 64

    # ── TUNING PARAMETERS (OPTIMIZED FOR LIFECYCLE DETECTION) ───────────────────
    ACTIVE_SMOOTHING_METHOD = "Savitzky-Golay"

    # Savitzky-Golay window tightened to 7 to let fast/initial spikes pop out bigger
    SAVGOL_WINDOW = 7
    SAVGOL_POLY = 3

    # Microscopic local floors to prevent empty videos from magnifying background noise
    SAFE_NOISE_FLOOR = {
        "Bhattacharyya": 0.005,
        "Cosine": 0.0005
    }

    # Advanced Anti-Noise Peak Constraints & Window Cutoffs
    PEAK_PROMINENCE_FACTOR = 0.12  # Threshold ratio letting smaller human events register ....Sensitivity multiplier: Peaks must stand out by 12% of total range to be captured
    MIN_PEAK_HEIGHT = 0.15         # Let smaller initial spikes clear the detector baseline ....Strict amplitude gate: Ignores low-level signal fluctuations under 15%
    MIN_PEAK_DISTANCE_SECS = 5     # Minimum running spacing required between distinct event ....Temporal spacing: Forces the detector to merge spikes occurring within 5 seconds of each other
    BASE_SLOPE_CUTOFF = 0.20       # Trace segment ends at 20% height above localized valley floor ....Segment boundary: Cuts off segment tracking at 20% altitude above local valley floors

    metric_colors = {"Bhattacharyya": "#d62728", "Cosine": "#1f77b4"}

    # ==============================================================================
    # ── SECTION 3: MAIN VIDEO PROCESSING & DATA BUFFERING LOOP ────────────────────
    # ==============================================================================
    all_videos_data = {}  # In-memory dictionary to store processed time-series outputs

    for idx, v_name in enumerate(video_paths, 1):
        emit(f"🎬 [Processing {idx}/{len(video_paths)}] Extracting raw distance vectors for: {v_name}", job_id=job_id,
             stage="segmentation", ui_message=f"Scanning {v_name} for scene changes ({idx}/{len(video_paths)})...")
        v_path = os.path.join(VIDEO_DIR, v_name)

        result = extract_video_metrics(
            v_path,
            active_smoothing_method=ACTIVE_SMOOTHING_METHOD,
            savgol_window=SAVGOL_WINDOW,
            savgol_poly=SAVGOL_POLY,
            bins=BINS,
            job_id=job_id  # threaded through so emit() inside knows which job's log to append to
        )

        if result is None:
            emit(f"⚠️ Skipping damaged/unopenable/too-short video file: {v_name}", job_id=job_id,
                 stage="segmentation", ui_message=f"Skipping {v_name} — couldn't process it.")
            continue

        timestamps, metrics_raw, metrics_filtered = result
        all_videos_data[v_name] = {
            "timestamps": timestamps,
            "metrics_raw": metrics_raw,
            "metrics_filtered": metrics_filtered
        }

    # ==============================================================================
    # ── SECTION 4: STANDALONE TOPOGRAPHICAL SEGMENTATION ENGINE ───────────────────
    # ==============================================================================
    excel_metadata_list = []

    for v_name, data in all_videos_data.items():
        timestamps = data["timestamps"]
        metrics_raw = data["metrics_raw"]
        metrics_filtered = data["metrics_filtered"]
        video_base_name = os.path.splitext(v_name)[0]

        emit(f"\n==========================================================", job_id=job_id,
             stage="segmentation", ui_message="")  # separator — terminal only, nothing worth showing in the UI
        emit(f"📋 REPORT SUMMARY FOR ASSET: {v_name}", job_id=job_id,
             stage="segmentation", ui_message=f"Finished scanning {v_name}.")
        emit(f"==========================================================", job_id=job_id,
             stage="segmentation", ui_message="")  # separator — terminal only, nothing worth showing in the UI

        for name, filtered_array in metrics_filtered.items():
            scaled_smoothed, scaled_raw, segments = scale_and_segment(
                filtered_array,
                metrics_raw[name],
                noise_floor=SAFE_NOISE_FLOOR[name],
                peak_prominence_factor=PEAK_PROMINENCE_FACTOR,
                min_peak_height=MIN_PEAK_HEIGHT,
                min_peak_distance_secs=MIN_PEAK_DISTANCE_SECS,
                base_slope_cutoff=BASE_SLOPE_CUTOFF
            )

            excel_rows = log_segments(v_name, name, segments, timestamps, job_id=job_id)
            excel_metadata_list.extend(excel_rows)

            out_file_name = f"final_segmented_{video_base_name}_{name.lower()}.png"
            render_segment_plot(
                v_name, name, timestamps, scaled_raw, scaled_smoothed, segments,
                color=metric_colors[name],
                output_path=os.path.join(SAVE_DIR, out_file_name),
                job_id=job_id
            )

    # ==============================================================================
    # ── SECTION 5: SMART NATIVE EXCEL APPENDING & DE-DUPLICATION ENGINE ───────────
    # ==============================================================================
    save_segment_excel(excel_metadata_list, EXCEL_PATH, job_id=job_id)

    emit("\n🎉 Complete! Sorted everything into perfectly isolated segments.", job_id=job_id,
         stage="segmentation", ui_message="Video segmentation complete.")


if __name__ == "__main__":
    # Local manual test: drop a video into storage/jobs/local_test/raw/ first,
    # then run this file directly to confirm the whole chain works end-to-end.
    main(job_id="local_test")