# -*- coding: utf-8 -*-
"""
Gemma_Slice_and_Excel_timestamps.py

This is where we SLICE THE MAIN VIDEO INTO SEGMENTS based on the calculations
it is again doing (yes there are redundancies of the same process... like
segmentation of timestamps and so on... which I will change and optimise
later) and save the segments in a folder.

REFACTOR NOTE: The long processing chunks (per-video Cosine extraction,
peak/segment detection, slicing+Excel logging, Excel merge) now live in
slice_pipeline_lib.py as callable functions. This file just holds your
config/tuning parameters and orchestrates the single processing loop by
calling into that library.

The whole thing is wrapped in main() so this file can either be run directly
(`python Gemma_Slice_and_Excel_timestamps.py`) or imported and called from a
higher-level orchestrator script, e.g.:
    import gemma_slice_and_excel_timestamps
    gemma_slice_and_excel_timestamps.main()

NOTE: Before running, install dependencies in your terminal:
    pip install opencv-python numpy scipy pandas openpyxl

Make sure slice_pipeline_lib.py is in the same folder as this script (or
somewhere on your PYTHONPATH) so the import below resolves.
"""

import os
import pandas as pd

from step_2_functions_video_slice_excel_timestamp import (
    extract_cosine_curve,
    find_cosine_segments,
    slice_video_segments,
    save_segment_excel,
)


def main():
    # 1. LOCAL DIRECTORY SETUP
    # 🔧 EDIT THIS to match wherever you keep the videos on your machine.
    MAIN_VIDEO_DIR = r"D:\Traffic_Control\trial_main_video"
    MASTER_DIR = r"D:\Traffic_Control"
    SEGMENT_DIR = os.path.join(MASTER_DIR, "trial_video_segments")
    EXCEL_PATH = os.path.join(MASTER_DIR, "segment_timestamps.xlsx")

    if not os.path.exists(MAIN_VIDEO_DIR):
        raise FileNotFoundError(f"⚠️ Could not find processing folder: {MAIN_VIDEO_DIR}.")

    os.makedirs(SEGMENT_DIR, exist_ok=True)  # Create the output folder for sliced clips if it doesn't exist yet

    all_files = os.listdir(MAIN_VIDEO_DIR)
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_paths = [f for f in all_files if f.lower().endswith(video_extensions)]

    print(f"📦 Found {len(video_paths)} videos. Starting Engine...")

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
        print(f"\n🎬 [Asset {idx}/{len(video_paths)}] Processing: {v_name}")

        result = extract_cosine_curve(v_path, savgol_window=SAVGOL_WINDOW, savgol_poly=SAVGOL_POLY, bins=BINS)
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
            print("  📍 No significant transitions found. Skipping.")
            continue

        excel_rows = slice_video_segments(v_path, v_name, segments, timestamps, fps, SEGMENT_DIR)
        excel_metadata_list.extend(excel_rows)

    # ==============================================================================
    # ── STEP 5: EXCEL SPREADSHEET CREATION (THE MAP FOR YOLOX) ────────────────────
    # ==============================================================================
    save_segment_excel(excel_metadata_list, EXCEL_PATH)

    print("🎉 PIPELINE RUN COMPLETE!")


if __name__ == "__main__":
    main()