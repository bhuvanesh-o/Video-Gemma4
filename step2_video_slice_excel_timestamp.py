# -*- coding: utf-8 -*-
"""
Gemma_Slice_and_Excel_timestamps.py

Ported from Google Colab (Gemma_Slice_and_Excel_timestamps.ipynb) to a
standalone local script for VS Code.

This is where we SLICE THE MAIN VIDEO INTO SEGMENTS based on the calculations
it is again doing (yes there are redundancies of the same process... like
segmentation of timestamps and so on... which I will change and optimise
later) and save the segments in a folder.

NOTE: Before running, install dependencies in your terminal:
    pip install opencv-python numpy scipy pandas openpyxl
"""

# ==============================================================================
# ── BRAIN 1: MACRO SLICER & EXCEL MAP GENERATOR ───────────────────────────────
# ==============================================================================

import cv2
import numpy as np
import os
import scipy.spatial.distance as dist
from scipy.signal import find_peaks, savgol_filter
import pandas as pd  # 🌟 CRITICAL: Pandas is here to make the Excel sheet!

# 1. LOCAL DIRECTORY SETUP
# Colab's drive.mount() is gone — these now point straight at local folders.
# 🔧 EDIT THIS to match wherever you keep the videos on your machine.
MASTER_DIR = r"C:\Users\YourName\Videos\GEMMA\trial_main_video"
SEGMENT_DIR = os.path.join(MASTER_DIR, "trial_video_segments")
EXCEL_PATH = os.path.join(MASTER_DIR, "segment_timestamps.xlsx")

if not os.path.exists(MASTER_DIR):
    raise FileNotFoundError(f"⚠️ Could not find processing folder: {MASTER_DIR}.")

os.makedirs(SEGMENT_DIR, exist_ok=True)  # Create the output folder for sliced clips if it doesn't exist yet

all_files = os.listdir(MASTER_DIR)
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


# 3. HELPER FUNCTIONS
def compute_histogram(frame, bins=BINS):
    """Convert a BGR frame to grayscale and return its normalized intensity histogram."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
    hist = hist / (np.sum(hist) + 1e-9)  # Normalize so the histogram sums to 1 (avoid divide-by-zero)
    return hist.flatten()


def slice_physical_mp4(input_path, start_sec, end_sec, output_path, fps):
    """
    Physically re-encode a sub-clip of the source video between start_sec and
    end_sec, writing it out as its own standalone .mp4 file.
    """
    cap = cv2.VideoCapture(input_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0, start_sec * 1000.0))  # Seek to the segment's start time

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Read and write frames one by one until we pass the segment's end time
    while cap.isOpened():
        current_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if current_sec > end_sec:
            break
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()


# ==============================================================================
# ── STEP 4: MASS PROCESSING LOOP ──────────────────────────────────────────────
# ==============================================================================

for idx, v_name in enumerate(video_paths, 1):
    v_path = os.path.join(MASTER_DIR, v_name)
    cap = cv2.VideoCapture(v_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Skip videos that fail to open or report a 0 FPS (corrupt/unreadable files)
    if not cap.isOpened() or fps == 0:
        continue
    print(f"\n🎬 [Asset {idx}/{len(video_paths)}] Processing: {v_name}")

    frames, timestamps = [], []
    frame_idx = 0

    # Sample exactly 1 frame per second to keep the change-detection pass fast
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % int(fps) == 0:
            frames.append(frame)
            timestamps.append(frame_idx / fps)
        frame_idx += 1
    cap.release()

    n_frames = len(frames)
    if n_frames < 2:
        continue  # Not enough frames to compute frame-to-frame change

    # Build per-second histograms, then measure cosine distance between consecutive frames
    histograms_norm = [compute_histogram(f) for f in frames]
    cosine_raw = np.zeros(n_frames)
    for i in range(1, n_frames):
        cosine_raw[i] = dist.cosine(histograms_norm[i - 1], histograms_norm[i])

    # Smooth the raw change curve with Savitzky-Golay to suppress frame-flicker noise
    if len(cosine_raw) > SAVGOL_WINDOW:
        cosine_smoothed = savgol_filter(cosine_raw, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLY)
    else:
        cosine_smoothed = np.copy(cosine_raw)

    # Normalize the smoothed curve to a 0-1 scale, with a noise floor safety guard
    local_max = np.max(cosine_smoothed)
    divisor = max(local_max, SAFE_NOISE_FLOOR)
    scaled_smoothed = cosine_smoothed / divisor
    curve_range = np.max(scaled_smoothed) - np.min(scaled_smoothed)

    # Run SciPy's peak detector with prominence/height/distance constraints tuned above
    detected_peaks, _ = find_peaks(
        scaled_smoothed,
        prominence=PEAK_PROMINENCE_FACTOR * curve_range,
        height=MIN_PEAK_HEIGHT,
        distance=MIN_PEAK_DISTANCE_SECS
    )
    peaks = sorted(list(set(detected_peaks)))

    # ── BOUNDARY EDGE SPIKE RECOVERY ──────────────────────────────────────────
    # Catches activity that's already happening right at the very start or end
    # of the video, which find_peaks can miss since it can't see past the edges.
    edge_frame_window = int(MIN_PEAK_DISTANCE_SECS)

    if len(scaled_smoothed) > edge_frame_window:
        front_max_idx = np.argmax(scaled_smoothed[:edge_frame_window])
        if scaled_smoothed[front_max_idx] >= MIN_PEAK_HEIGHT:
            if not any(abs(front_max_idx - p) < MIN_PEAK_DISTANCE_SECS for p in peaks):
                peaks.insert(0, front_max_idx)

    if len(scaled_smoothed) > edge_frame_window:
        back_start_idx = len(scaled_smoothed) - edge_frame_window
        back_max_idx = back_start_idx + np.argmax(scaled_smoothed[back_start_idx:])
        if scaled_smoothed[back_max_idx] >= MIN_PEAK_HEIGHT:
            if not any(abs(back_max_idx - p) < MIN_PEAK_DISTANCE_SECS for p in peaks):
                peaks.append(back_max_idx)

    peaks = sorted(list(set(peaks)))
    num_peaks = len(peaks)
    segments = []

    # ── SADDLE-POINT SEGMENT SPLITTING ────────────────────────────────────────
    # For each peak, walk down its left/right slopes toward the nearest valley
    # to find where the activity actually starts and ends (not just the peak).
    for i in range(num_peaks):
        p_idx = peaks[i]
        left_limit_idx = 0 if i == 0 else peaks[i - 1]
        right_limit_idx = (len(scaled_smoothed) - 1) if i == (num_peaks - 1) else peaks[i + 1]

        left_valley_idx = left_limit_idx + np.argmin(scaled_smoothed[left_limit_idx:p_idx + 1])
        right_valley_idx = p_idx + np.argmin(scaled_smoothed[p_idx:right_limit_idx + 1])

        v_peak = scaled_smoothed[p_idx]
        v_left_floor = scaled_smoothed[left_valley_idx]
        v_right_floor = scaled_smoothed[right_valley_idx]

        # Cutoff sits 20% of the way up from each valley floor toward the peak
        thresh_l = v_left_floor + BASE_SLOPE_CUTOFF * (v_peak - v_left_floor)
        thresh_r = v_right_floor + BASE_SLOPE_CUTOFF * (v_peak - v_right_floor)

        start_idx = p_idx
        while start_idx > left_valley_idx and scaled_smoothed[start_idx] > thresh_l:
            start_idx -= 1

        end_idx = p_idx
        while end_idx < right_valley_idx and scaled_smoothed[end_idx] > thresh_r:
            end_idx += 1

        segments.append({'start_idx': start_idx, 'peak_idx': p_idx, 'end_idx': end_idx})

    if not segments:
        print("  📍 No significant transitions found. Skipping.")
        continue

    # Make a per-video subfolder to hold its sliced segment clips
    video_base_name = os.path.splitext(v_name)[0]
    video_output_dir = os.path.join(SEGMENT_DIR, video_base_name)
    os.makedirs(video_output_dir, exist_ok=True)

    for seg_idx, seg in enumerate(segments, 1):
        t_start = timestamps[seg['start_idx']]
        t_end = timestamps[seg['end_idx']]

        segment_clip_name = f"segment_{seg_idx}.mp4"
        segment_clip_path = os.path.join(video_output_dir, segment_clip_name)

        print(f"    ✂️ Slicing -> {segment_clip_name} ({t_start:.1f}s to {t_end:.1f}s)...")
        slice_physical_mp4(v_path, t_start, t_end, segment_clip_path, fps)

        # 🌟 CRITICAL: Saving the data to the memory list for Excel
        excel_metadata_list.append({
            "Source_Video": v_name,
            "Segment_ID": seg_idx,
            "Start_Time": round(t_start, 2),
            "End_Time": round(t_end, 2)
        })

# ==============================================================================
# ── STEP 5: EXCEL SPREADSHEET CREATION (THE MAP FOR YOLOX) ────────────────────
# ==============================================================================
if len(excel_metadata_list) > 0:
    df_new = pd.DataFrame(excel_metadata_list)

    # Merge with any pre-existing workbook, de-duplicating re-scanned segments
    if os.path.exists(EXCEL_PATH):
        try:
            df_historical = pd.read_excel(EXCEL_PATH)
            df_combined = pd.concat([df_historical, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=["Source_Video", "Segment_ID", "Start_Time"], keep="last")
        except Exception:
            df_combined = df_new
    else:
        df_combined = df_new

    df_combined.to_excel(EXCEL_PATH, index=False)
    print(f"\n✅ SUCCESS! Master timeline saved to:\n➡️ {EXCEL_PATH}")
else:
    print("\n⚠️ No movement detected across any videos. Excel sheet was not created.")

print("🎉 PIPELINE RUN COMPLETE!")