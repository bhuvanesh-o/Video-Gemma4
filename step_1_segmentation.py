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
(`python step_1_segmentation.py.py`) or imported and called from a higher-level
orchestrator script, e.g.:
    import step_1_segmentation.py
    step_1_segmentation.py.main()

NOTE: Before running, install dependencies in your terminal:
    pip install opencv-python numpy matplotlib scipy pandas openpyxl

Make sure step_1_functions_segmentation.py is in the same folder as this script (or
somewhere on your PYTHONPATH) so the import below resolves.
"""

import os
import numpy as np
import pandas as pd

from step_1_functions_segmentation import (
    extract_video_metrics,
    scale_and_segment,
    log_segments,
    render_segment_plot,
    save_segment_excel,
)


def main():
    # ── LOCAL PATH CONFIG ─────────────────────────────────────────────────────
    # Use raw strings (r"...") on Windows so backslashes don't get treated as escape characters.
    VIDEO_DIR = r"D:\Traffic_Control\trial_main_video"  # 🔧 EDIT THIS
    SAVE_DIR = r"D:\Traffic_Control"   # 🔧 EDIT THIS

    if not os.path.exists(VIDEO_DIR):
        raise FileNotFoundError(f"⚠️ Could not find folder: {VIDEO_DIR}. Check your path spelling!")

    all_files = os.listdir(VIDEO_DIR)
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_paths = [f for f in all_files if f.lower().endswith(video_extensions)]

    print(f"📦 Found {len(video_paths)} videos. Ingesting mass-processing engine...")

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
        print(f"🎬 [Processing {idx}/{len(video_paths)}] Extracting raw distance vectors for: {v_name}")
        v_path = os.path.join(VIDEO_DIR, v_name)

        result = extract_video_metrics(
            v_path,
            active_smoothing_method=ACTIVE_SMOOTHING_METHOD,
            savgol_window=SAVGOL_WINDOW,
            savgol_poly=SAVGOL_POLY,
            bins=BINS
        )

        if result is None:
            print(f"⚠️ Skipping damaged/unopenable/too-short video file: {v_name}")
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

        print(f"\n==========================================================")
        print(f"📋 REPORT SUMMARY FOR ASSET: {v_name}")
        print(f"==========================================================")

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

            excel_rows = log_segments(v_name, name, segments, timestamps)
            excel_metadata_list.extend(excel_rows)

            out_file_name = f"final_segmented_{video_base_name}_{name.lower()}.png"
            render_segment_plot(
                v_name, name, timestamps, scaled_raw, scaled_smoothed, segments,
                color=metric_colors[name],
                output_path=os.path.join(SAVE_DIR, out_file_name)
            )

    # ==============================================================================
    # ── SECTION 5: SMART NATIVE EXCEL APPENDING & DE-DUPLICATION ENGINE ───────────
    # ==============================================================================
    save_segment_excel(excel_metadata_list, os.path.join(SAVE_DIR, "segment_timestamps.xlsx"))

    print("\n🎉 Complete! Sorted everything into perfectly isolated segments.")


if __name__ == "__main__":
    main()
