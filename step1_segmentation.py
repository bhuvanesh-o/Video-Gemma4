# -*- coding: utf-8 -*-
"""
Gemma_Segment.py

Ported from Google Colab (Gemma_Segment.ipynb) to a standalone local script
for VS Code. Logic and comments are unchanged from the notebook version —
only Colab-specific bits (pip magic command, Drive mount) were removed/adapted
for local execution.

We are looking through the MAIN VIDEO and plotting the CHANGE POINT GRAPH and
we are updating all the SEGMENT TIMES INTO EXCEL SHEET and we will also print
the timings of each segment in the O/P also.

NOTE: Before running, install dependencies in your terminal:
    pip install opencv-python numpy matplotlib scipy pandas openpyxl
"""

# ── SECTION 1: Master Setup, Imports & Configuration ────────────────────────
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.spatial.distance as dist
from scipy.ndimage import grey_opening
from scipy.signal import find_peaks, savgol_filter
import pandas as pd  # ── 🌟 UPDATE PART: Imported Pandas for native Excel creation

# ── LOCAL PATH CONFIG ─────────────────────────────────────────────────────
# Colab's drive.mount() is gone — point these straight at your local folders
# (e.g. on your Ryzen/HP machine). Use raw strings (r"...") on Windows so
# backslashes don't get treated as escape characters.
VIDEO_DIR = r"D:\Traffic_Control\trial_main_video"  # 🔧 EDIT THIS
SAVE_DIR = r"C:\D:\Traffic_Control"   # 🔧 EDIT THIS

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


# ==============================================================================
# ── SECTION 2: ADVANCED SMOOTHING MATHEMATICAL ALGORITHMS ─────────────────────
# ==============================================================================

def compute_histogram(frame, bins=BINS):
    """
    Grayscale Color Space Converter & Histogram Extractor.
    Normalizes pixel counts into a continuous probability distribution between 0.0 and 1.0.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
    hist = hist / (np.sum(hist) + 1e-9)  # Prevent divide-by-zero errors on empty/black frames
    return hist.flatten()


# ==============================================================================
# ── SECTION 3: MAIN VIDEO PROCESSING & DATA BUFFERING LOOP ────────────────────
# ==============================================================================
all_videos_data = {}  # In-memory dictionary to store processed time-series outputs

for idx, v_name in enumerate(video_paths, 1):
    print(f"🎬 [Processing {idx}/{len(video_paths)}] Extracting raw distance vectors for: {v_name}")
    v_path = os.path.join(VIDEO_DIR, v_name)
    cap = cv2.VideoCapture(v_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Error handling guard: Verify video validity and confirm frame rates match standard capture parameters
    if not cap.isOpened() or fps == 0:
        print(f"⚠️ Skipping damaged or unopenable video file: {v_name}")
        continue

    frames, timestamps = [], []
    frame_idx = 0

    # ── FRAME EXTRACTION LAYER ────────────────────────────────────────────────
    # Decodes video files frame-by-frame, extracting exactly 1 frame per second to speed up processing
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video stream reached
        if frame_idx % int(fps) == 0:
            frames.append(frame)
            timestamps.append(frame_idx / fps)
        frame_idx += 1
    cap.release()  # Close file descriptor back to operational memory system

    n_frames = len(frames)
    if n_frames < 2:
        continue  # Skip files that don't have enough frames for comparative step math

    # Convert extracted image arrays into flattened structural probability histograms
    histograms_norm = [compute_histogram(f) for f in frames]

    # Instantiate zeroed array tracks to store raw mathematical change vectors
    metrics_raw = {
        "Bhattacharyya": np.zeros(n_frames),
        "Cosine": np.zeros(n_frames)
    }

    # ── DISTANCE CALCULATION LOOP ─────────────────────────────────────────────
    # Computes step-by-step vector variances between consecutive frames
    for i in range(1, n_frames):
        P = histograms_norm[i - 1]  # Frame representation at time t - 1
        Q = histograms_norm[i]      # Frame representation at time t

        # Compare overlapping profile shifts using alternative geometric logic equations
        metrics_raw["Bhattacharyya"][i] = cv2.compareHist(P, Q, cv2.HISTCMP_BHATTACHARYYA)
        metrics_raw["Cosine"][i] = dist.cosine(P, Q)

    # ── TEMPORAL FILTER EXECUTOR ──────────────────────────────────────────────
    # Feeds raw distance vectors into the selected smoothing engine to suppress frame-flicker noise
    metrics_filtered = {}
    for name, raw_array in metrics_raw.items():

        '''
        
        if ACTIVE_SMOOTHING_METHOD == "Bilateral":
            smoothed_data = bilateral_filter1d(raw_array, spatial_window=11, sigma_intensity=0.02)
        elif ACTIVE_SMOOTHING_METHOD == "Top-Hat":
            smoothed_data = top_hat_baseline_suppression(raw_array, window_size=15)

        '''

        if ACTIVE_SMOOTHING_METHOD == "Savitzky-Golay":
            # Guard logic: Ensure the dataset contains more frames than the smoothing window size
            if len(raw_array) > SAVGOL_WINDOW:
                smoothed_data = savgol_filter(raw_array, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLY)
            else:
                smoothed_data = np.copy(raw_array)
        else:
            smoothed_data = np.copy(raw_array)  # Pass-through fallback configuration

        metrics_filtered[name] = smoothed_data

    # Map processed tracks to storage key arrays linked to the file name
    all_videos_data[v_name] = {
        "timestamps": np.array(timestamps),
        "metrics_raw": metrics_raw,
        "metrics_filtered": metrics_filtered
    }


# ==============================================================================
# ── SECTION 4: STANDALONE TOPOGRAPHICAL SEGMENTATION ENGINE ───────────────────
# ==============================================================================
metric_colors = {"Bhattacharyya": "#d62728", "Cosine": "#1f77b4"}

# Create an empty memory buffer list to store our row logs before starting the loop
excel_metadata_list = []

for v_name, data in all_videos_data.items():
    timestamps = data["timestamps"]
    metrics_raw = data["metrics_raw"]
    metrics_filtered = data["metrics_filtered"]
    video_base_name = os.path.splitext(v_name)[0]

    print(f"\n==========================================================")
    print(f"📋 LLM PIPELINE REPORT SUMMARY FOR ASSET: {v_name}")
    print(f"==========================================================")

    for name, filtered_array in metrics_filtered.items():
        # Global Dataset Normalization: Scales the data between 0 and 1 using a noise ceiling safety guard
        local_max = np.max(filtered_array)
        divisor = max(local_max, SAFE_NOISE_FLOOR[name])

        scaled_smoothed = filtered_array / divisor
        scaled_raw = metrics_raw[name] / divisor

        # Calculate the dynamic curve range to optimize peak prominence checks
        curve_range = np.max(scaled_smoothed) - np.min(scaled_smoothed)

        # 1. Run standard internal prominence search using SciPy's geometric peak locator
        detected_peaks, _ = find_peaks(
            scaled_smoothed,
            prominence=PEAK_PROMINENCE_FACTOR * curve_range,
            height=MIN_PEAK_HEIGHT,
            distance=MIN_PEAK_DISTANCE_SECS
        )
        peaks = list(detected_peaks)

        # ── RECOVERY LAYER: BOUNDARY EDGE SPIKE RECOVERY ────────────────────
        # Guards against missing cuts if a vehicle is already present at 0.0s or the final frame
        edge_frame_window = int(MIN_PEAK_DISTANCE_SECS)

        # A. Check Front Edge Boundary (Start of Video)
        if len(scaled_smoothed) > edge_frame_window:
            front_max_idx = np.argmax(scaled_smoothed[:edge_frame_window])
            if scaled_smoothed[front_max_idx] >= MIN_PEAK_HEIGHT:
                # Insert peak if no other detected peaks are within the minimum distance window
                if not any(abs(front_max_idx - p) < MIN_PEAK_DISTANCE_SECS for p in peaks):
                    peaks.insert(0, front_max_idx)

        # B. Check Back Edge Boundary (End of Video)
        if len(scaled_smoothed) > edge_frame_window:
            back_start_idx = len(scaled_smoothed) - edge_frame_window
            back_max_idx = back_start_idx + np.argmax(scaled_smoothed[back_start_idx:])
            if scaled_smoothed[back_max_idx] >= MIN_PEAK_HEIGHT:
                # Append peak if no other detected peaks are within the minimum distance window
                if not any(abs(back_max_idx - p) < MIN_PEAK_DISTANCE_SECS for p in peaks):
                    peaks.append(back_max_idx)

        # Ensure unique array indexing elements sorted chronologically
        peaks = sorted(list(set(peaks)))
        num_peaks = len(peaks)
        # ────────────────────────────────────────────────────────────────────

        segments = []

        # ── ADVANCED TOPOGRAPHICAL NEIGHBOR SADDLE-POINT SPLIT ENGINE ─────────
        # Traces activity lifecycles by walking down the hills of each peak to locate the true motion onset and offset
        for i in range(num_peaks):
            p_idx = peaks[i]

            # Set localized boundary constraints using neighboring peaks as hard cutoffs
            left_limit_idx = 0 if i == 0 else peaks[i - 1]
            right_limit_idx = (len(scaled_smoothed) - 1) if i == (num_peaks - 1) else peaks[i + 1]

            # Identify local minimum valleys (the saddle intersections between events)
            left_valley_idx = left_limit_idx + np.argmin(scaled_smoothed[left_limit_idx:p_idx + 1])
            right_valley_idx = p_idx + np.argmin(scaled_smoothed[p_idx:right_limit_idx + 1])

            v_peak = scaled_smoothed[p_idx]
            v_left_floor = scaled_smoothed[left_valley_idx]
            v_right_floor = scaled_smoothed[right_valley_idx]

            # Compute custom relative cutoff parameters for each slope independently (20% above valley floor)
            thresh_l = v_left_floor + BASE_SLOPE_CUTOFF * (v_peak - v_left_floor)
            thresh_r = v_right_floor + BASE_SLOPE_CUTOFF * (v_peak - v_right_floor)

            # Walk backward down the left slope to find where the activity begins
            start_idx = p_idx
            while start_idx > left_valley_idx and scaled_smoothed[start_idx] > thresh_l:
                start_idx -= 1

            # Walk forward down the right slope to find where the activity settles down
            end_idx = p_idx
            while end_idx < right_valley_idx and scaled_smoothed[end_idx] > thresh_r:
                end_idx += 1

            # Append the calculated frame indices to the segment tracking block
            segments.append({
                'start_idx': start_idx,
                'peak_idx': p_idx,
                'end_idx': end_idx
            })

        # ── RENDERING & VISUALIZATION LAYER ───────────────────────────────────
        fig, ax = plt.subplots(figsize=(20, 7))

        # Draw background raw trace line with a faint opacity to keep the plot scannable
        ax.plot(timestamps, scaled_raw, color=metric_colors[name], alpha=0.15, label=f"{name} (Raw)")
        # Draw solid smoothed trend line over the raw trace data
        ax.plot(timestamps, scaled_smoothed, color=metric_colors[name], linewidth=2.2, label=f"{name} (Smoothed)")

        print(f"\n🔹 Distance Metric Model Integration: {name}")

        # Render sequential shaded segments and map directly to console outputs
        for seg_idx, seg in enumerate(segments, 1):
            t_start = timestamps[seg['start_idx']]
            t_peak = timestamps[seg['peak_idx']]
            t_end = timestamps[seg['end_idx']]

            # Output pure structured timestamp data to the console for quick reference
            print(f"  📍 Segment {seg_idx} Window Details -> Start: {t_start:.1f}s | Max Peak: {t_peak:.1f}s | Finish: {t_end:.1f}s")

            # EXCEL TRACKING INJECTION: Log only the Cosine metrics to avoid creating duplicate rows,
            # as the downstream object detection pipeline tracks along your Cosine graph timelines.
            if name == "Cosine":
                excel_metadata_list.append({
                    "Source_Video": v_name,
                    "Segment_ID": seg_idx,
                    "Start_Time": round(t_start, 2),
                    "End_Time": round(t_end, 2)
                })

            # Shade the active segment window container area with a light grey background
            ax.axvspan(t_start, t_end, color='gray', alpha=0.06)

            # Draw vertical transition markers: Green (Rise/Start), Blue (Peak), Red (Fall/Finish)
            ax.axvline(t_start, color='#2ca02c', linestyle='--', linewidth=1.5, alpha=0.85)
            ax.axvline(t_peak, color='#1f77b4', linestyle='-.', linewidth=1.5, alpha=0.85)
            ax.axvline(t_end, color='#d62728', linestyle=':', linewidth=1.5, alpha=0.85)

            # Render text labels at the top of each shaded block
            ax.text((t_start + t_end) / 2, 1.02, f"SEGMENT {seg_idx}", color='black', weight='bold',
                    fontsize=9, horizontalalignment='center', verticalalignment='bottom',
                    bbox=dict(facecolor='#f8f9fa', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.3'))

            # Stagger local label boxes vertically to prevent text overlapping if events occur close together
            y_box = 0.85 if seg_idx % 2 == 0 else 0.68
            ax.text(t_start, y_box, f"Rise: {t_start:.1f}s", color='green', weight='bold', fontsize=8,
                    verticalalignment='center', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='green', boxstyle='round,pad=0.2'))
            ax.text(t_end, y_box, f"Fall: {t_end:.1f}s", color='red', weight='bold', fontsize=8,
                    verticalalignment='center', horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='red', boxstyle='round,pad=0.2'))

        # Visual Canvas Axis Tuning
        plt.ylim(-0.1, 1.1)  # Extra padding prevents the bounding boxes at the top and bottom from being clipped
        plt.ylabel("Adaptive Normalized Scale [0-1]", fontsize=10, weight='bold')
        plt.xlabel("Time (Seconds)", fontsize=11)
        plt.title(f"Standalone Clean Segment Axis Panel: {name}", fontsize=11, weight='bold', loc='left', pad=4)
        plt.grid(True, linestyle='--', alpha=0.35)
        plt.legend(loc='upper right', framealpha=0.95, fontsize=9)

        plt.suptitle(f"Strict Ordered Segment Report \nSource File: {v_name}", fontsize=12, weight='bold', y=0.98)
        plt.tight_layout()

        # Save high-resolution PNG copies directly to the specified folder (locally, instead of Drive)
        out_file_name = f"final_segmented_{video_base_name}_{name.lower()}.png"
        plt.savefig(os.path.join(SAVE_DIR, out_file_name), dpi=150)
        plt.show()  # Display the plot in a local matplotlib window (no notebook inline rendering on VS Code)

# ==============================================================================
# ── SECTION 5: SMART NATIVE EXCEL APPENDING & DE-DUPLICATION ENGINE ───────────
# ==============================================================================
if len(excel_metadata_list) > 0:
    # Convert today's fresh runs into a structured DataFrame
    df_new = pd.DataFrame(excel_metadata_list)
    excel_output_path = os.path.join(SAVE_DIR, "segment_timestamps.xlsx")

    # Check for an existing database workbook file in the folder to merge new records safely
    if os.path.exists(excel_output_path):
        print("\n📂 Found existing historical segment workbook. Merging new streams...")
        try:
            # Read the historical data sheet
            df_historical = pd.read_excel(excel_output_path)

            # Stack new data rows directly underneath historical rows
            df_combined = pd.concat([df_historical, df_new], ignore_index=True)

            # DE-DUPLICATION GUARD: If a video is re-scanned, drop its old records and keep the latest ones
            df_combined = df_combined.drop_duplicates(subset=["Source_Video", "Segment_ID", "Start_Time"], keep="last")
            df_combined = df_combined.reset_index(drop=True)
        except Exception as e:
            print(f"⚠️ Error reading old workbook file safely ({e}). Creating a fresh master block.")
            df_combined = df_new
    else:
        print("\n🆕 No historical database found. Creating a fresh master segment workbook...")
        df_combined = df_new

    # Write the cleaned data back to the binary Excel storage file
    df_combined.to_excel(excel_output_path, index=False)
    print(f"✅ SUCCESS! Programmatically committed timeline maps to master workbook directly:\n➡️ {excel_output_path}")
else:
    print("\n⚠️ Notification: No valid timeline changes triggered data logs.")

print("\n🎉 Complete! Sorted everything into perfectly isolated segments.")