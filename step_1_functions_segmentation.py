# -*- coding: utf-8 -*-
"""
step_1_functions_segmentation.py

Reusable functions pulled out of step_1_segmentation.py so they can be imported
and called from anywhere instead of living inline in one long script.

Each function below corresponds to one of the original "long chunks":
  - compute_histogram        <- Section 2 (unchanged)
  - extract_video_metrics    <- Section 3's per-video body (frame extraction,
                                 histogram distances, smoothing)
  - scale_and_segment        <- Section 4's normalization + peak-finding +
                                 saddle-point segment splitting
  - log_segments             <- Section 4's console report + Excel row logging
  - render_segment_plot      <- Section 4's matplotlib rendering/visualization
  - save_segment_excel       <- Section 5's smart append/de-duplication engine

REFACTOR NOTE: All print() calls have been replaced with emit() from progress.py
so that progress can be streamed to a browser later via FastAPI SSE, while still
printing locally when running from the command line. job_id=None is threaded
through every function that emits — passing None keeps local CLI behavior identical.
plt.show() is removed from render_segment_plot (a server has no display to pop a
window on — plt.savefig() + plt.close() is the correct server-side equivalent).
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.spatial.distance as dist
from scipy.signal import find_peaks, savgol_filter
import pandas as pd

from progress import emit


# ==============================================================================
# ── compute_histogram ──────────────────────────────────────────────────────────
# ==============================================================================
def compute_histogram(frame, bins=64):
    """
    Grayscale Color Space Converter & Histogram Extractor.
    Normalizes pixel counts into a continuous probability distribution between 0.0 and 1.0.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
    hist = hist / (np.sum(hist) + 1e-9)  # Prevent divide-by-zero errors on empty/black frames
    return hist.flatten()


# ==============================================================================
# ── extract_video_metrics ─────────────────────────────────────────────────────
# ==============================================================================
def extract_video_metrics(video_path, active_smoothing_method="Savitzky-Golay",
                           savgol_window=7, savgol_poly=3, bins=64, job_id=None):
    """
    Per-video body of the old Section 3 loop:
      1. Opens the video and validates it (fps/isOpened guard).
      2. Extracts exactly 1 frame per second.
      3. Computes Bhattacharyya + Cosine histogram-distance vectors between consecutive frames.
      4. Smooths both raw distance vectors with the selected method (currently only Savitzky-Golay
         is active — Bilateral/Top-Hat are left commented out below, matching the trimmed-down
         pipeline you're running).

    Returns (timestamps, metrics_raw, metrics_filtered) as a tuple, or None if the video
    is damaged/unopenable or doesn't have enough frames to compare.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Error handling guard: Verify video validity and confirm frame rates match standard capture parameters
    if not cap.isOpened() or fps == 0:
        cap.release()
        emit(f"⚠️ Could not open video or invalid FPS: {video_path}", job_id=job_id)
        return None

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
        emit(f"⚠️ Not enough frames to compare in: {video_path}", job_id=job_id)
        return None  # Not enough frames for comparative step math

    # Convert extracted image arrays into flattened structural probability histograms
    histograms_norm = [compute_histogram(f, bins) for f in frames]

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

        if active_smoothing_method == "Bilateral":
            smoothed_data = bilateral_filter1d(raw_array, spatial_window=11, sigma_intensity=0.02)
        elif active_smoothing_method == "Top-Hat":
            smoothed_data = top_hat_baseline_suppression(raw_array, window_size=15)

        '''

        if active_smoothing_method == "Savitzky-Golay":
            # Guard logic: Ensure the dataset contains more frames than the smoothing window size
            if len(raw_array) > savgol_window:
                smoothed_data = savgol_filter(raw_array, window_length=savgol_window, polyorder=savgol_poly)
            else:
                smoothed_data = np.copy(raw_array)
        else:
            smoothed_data = np.copy(raw_array)  # Pass-through fallback configuration

        metrics_filtered[name] = smoothed_data

    return np.array(timestamps), metrics_raw, metrics_filtered


# ==============================================================================
# ── scale_and_segment ─────────────────────────────────────────────────────────
# ==============================================================================
def scale_and_segment(filtered_array, raw_array, noise_floor,
                       peak_prominence_factor, min_peak_height,
                       min_peak_distance_secs, base_slope_cutoff):
    """
    Per-metric body of the old Section 4 loop, up to (but not including) plotting:
      1. Normalizes the filtered/raw curves to a 0-1 scale using a noise-floor safety guard.
      2. Runs SciPy's peak detector with your prominence/height/distance constraints.
      3. Recovers peaks sitting right at the very start/end edges of the video that
         find_peaks can't see past.
      4. Walks down each peak's left/right slopes to its saddle-point valleys to find
         the true motion onset/offset (the "ADVANCED TOPOGRAPHICAL... SPLIT ENGINE").

    Returns (scaled_smoothed, scaled_raw, segments) where segments is a list of dicts
    with 'start_idx', 'peak_idx', 'end_idx'.

    No emit() calls in here — this is pure math, no status messages needed.
    """
    # Global Dataset Normalization: Scales the data between 0 and 1 using a noise ceiling safety guard
    local_max = np.max(filtered_array)
    divisor = max(local_max, noise_floor)

    scaled_smoothed = filtered_array / divisor
    scaled_raw = raw_array / divisor

    # Calculate the dynamic curve range to optimize peak prominence checks
    curve_range = np.max(scaled_smoothed) - np.min(scaled_smoothed)

    # 1. Run standard internal prominence search using SciPy's geometric peak locator
    detected_peaks, _ = find_peaks(
        scaled_smoothed,
        prominence=peak_prominence_factor * curve_range,
        height=min_peak_height,
        distance=min_peak_distance_secs
    )
    peaks = list(detected_peaks)

    # ── RECOVERY LAYER: BOUNDARY EDGE SPIKE RECOVERY ────────────────────
    # Guards against missing cuts if a vehicle is already present at 0.0s or the final frame
    edge_frame_window = int(min_peak_distance_secs)

    # A. Check Front Edge Boundary (Start of Video)
    if len(scaled_smoothed) > edge_frame_window:
        front_max_idx = np.argmax(scaled_smoothed[:edge_frame_window])
        if scaled_smoothed[front_max_idx] >= min_peak_height:
            # Insert peak if no other detected peaks are within the minimum distance window
            if not any(abs(front_max_idx - p) < min_peak_distance_secs for p in peaks):
                peaks.insert(0, front_max_idx)

    # B. Check Back Edge Boundary (End of Video)
    if len(scaled_smoothed) > edge_frame_window:
        back_start_idx = len(scaled_smoothed) - edge_frame_window
        back_max_idx = back_start_idx + np.argmax(scaled_smoothed[back_start_idx:])
        if scaled_smoothed[back_max_idx] >= min_peak_height:
            # Append peak if no other detected peaks are within the minimum distance window
            if not any(abs(back_max_idx - p) < min_peak_distance_secs for p in peaks):
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
        thresh_l = v_left_floor + base_slope_cutoff * (v_peak - v_left_floor)
        thresh_r = v_right_floor + base_slope_cutoff * (v_peak - v_right_floor)

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

    return scaled_smoothed, scaled_raw, segments


# ==============================================================================
# ── log_segments ──────────────────────────────────────────────────────────────
# ==============================================================================
def log_segments(v_name, metric_name, segments, timestamps, job_id=None):
    """
    Console report + Excel row logging for one video/metric's segments.

    Prints the same "📍 Segment N Window Details" lines as the original script,
    and returns a list of row dicts ready to extend into excel_metadata_list —
    but ONLY for the Cosine metric (matching the original's "log only Cosine to
    avoid duplicate rows" comment). Other metrics return an empty list.
    """
    emit(f"\n🔹 Distance Metric Model Integration: {metric_name}", job_id=job_id)

    excel_rows = []
    for seg_idx, seg in enumerate(segments, 1):
        t_start = timestamps[seg['start_idx']]
        t_peak = timestamps[seg['peak_idx']]
        t_end = timestamps[seg['end_idx']]

        # Output pure structured timestamp data to the console for quick reference
        emit(f"  📍 Segment {seg_idx} Window Details -> Start: {t_start:.1f}s | Max Peak: {t_peak:.1f}s | Finish: {t_end:.1f}s", job_id=job_id)

        # EXCEL TRACKING INJECTION: Log only the Cosine metrics to avoid creating duplicate rows,
        # as the downstream object detection pipeline tracks along your Cosine graph timelines.
        if metric_name == "Cosine":
            excel_rows.append({
                "Source_Video": v_name,
                "Segment_ID": seg_idx,
                "Start_Time": round(t_start, 2),
                "End_Time": round(t_end, 2)
            })

    return excel_rows


# ==============================================================================
# ── render_segment_plot ───────────────────────────────────────────────────────
# ==============================================================================
def render_segment_plot(v_name, metric_name, timestamps, scaled_raw, scaled_smoothed,
                         segments, color, output_path, job_id=None):
    """
    Matplotlib RENDERING & VISUALIZATION LAYER from the original Section 4: draws the
    raw/smoothed traces, shades each segment window, marks rise/peak/fall lines, labels
    everything, and saves a high-res PNG to output_path.

    NOTE: plt.show() removed — a server has no display to pop a window on, and leaving
    it in would hang the background job indefinitely. plt.close(fig) is added to prevent
    matplotlib memory buildup across many jobs running back-to-back.
    """
    fig, ax = plt.subplots(figsize=(20, 7))

    # Draw background raw trace line with a faint opacity to keep the plot scannable
    ax.plot(timestamps, scaled_raw, color=color, alpha=0.15, label=f"{metric_name} (Raw)")
    # Draw solid smoothed trend line over the raw trace data
    ax.plot(timestamps, scaled_smoothed, color=color, linewidth=2.2, label=f"{metric_name} (Smoothed)")

    # Render sequential shaded segments and map directly to console outputs
    for seg_idx, seg in enumerate(segments, 1):
        t_start = timestamps[seg['start_idx']]
        t_peak = timestamps[seg['peak_idx']]
        t_end = timestamps[seg['end_idx']]

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
    plt.title(f"Standalone Clean Segment Axis Panel: {metric_name}", fontsize=11, weight='bold', loc='left', pad=4)
    plt.grid(True, linestyle='--', alpha=0.35)
    plt.legend(loc='upper right', framealpha=0.95, fontsize=9)

    plt.suptitle(f"Strict Ordered Segment Report \nSource File: {v_name}", fontsize=12, weight='bold', y=0.98)
    plt.tight_layout()

    # Save high-resolution PNG copies directly to the specified job folder
    plt.savefig(output_path, dpi=150)
    plt.close(fig)  # Close instead of show() — no display on a server; also avoids memory buildup across many jobs

    emit(f"🖼️ Saved segmentation plot -> {output_path}", job_id=job_id)


# ==============================================================================
# ── save_segment_excel ────────────────────────────────────────────────────────
# ==============================================================================
def save_segment_excel(excel_metadata_list, excel_output_path, job_id=None):
    """
    SMART NATIVE EXCEL APPENDING & DE-DUPLICATION ENGINE from the original Section 5.
    Merges today's fresh segment rows into any existing workbook, de-duplicating by
    (Source_Video, Segment_ID, Start_Time), and writes the result back out.
    """
    if len(excel_metadata_list) == 0:
        emit("\n⚠️ Notification: No valid timeline changes triggered data logs.", job_id=job_id)
        return

    # Convert today's fresh runs into a structured DataFrame
    df_new = pd.DataFrame(excel_metadata_list)

    # Check for an existing database workbook file in the folder to merge new records safely
    if os.path.exists(excel_output_path):
        emit("\n📂 Found existing historical segment workbook. Merging new streams...", job_id=job_id)
        try:
            # Read the historical data sheet
            df_historical = pd.read_excel(excel_output_path)

            # Stack new data rows directly underneath historical rows
            df_combined = pd.concat([df_historical, df_new], ignore_index=True)

            # DE-DUPLICATION GUARD: If a video is re-scanned, drop its old records and keep the latest ones
            df_combined = df_combined.drop_duplicates(subset=["Source_Video", "Segment_ID", "Start_Time"], keep="last")
            df_combined = df_combined.reset_index(drop=True)
        except Exception as e:
            emit(f"⚠️ Error reading old workbook file safely ({e}). Creating a fresh master block.", job_id=job_id)
            df_combined = df_new
    else:
        emit("\n🆕 No historical database found. Creating a fresh master segment workbook...", job_id=job_id)
        df_combined = df_new

    # Write the cleaned data back to the binary Excel storage file
    df_combined.to_excel(excel_output_path, index=False)
    emit(f"✅ SUCCESS! Programmatically committed timeline maps to master workbook directly:\n➡️ {excel_output_path}", job_id=job_id)