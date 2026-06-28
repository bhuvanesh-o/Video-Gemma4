# -*- coding: utf-8 -*-
"""
slice_pipeline_lib.py

Reusable functions pulled out of Gemma_Slice_and_Excel_timestamps.py.

  - compute_histogram      <- unchanged helper
  - extract_cosine_curve   <- per-video frame sampling + Cosine distance + smoothing
  - find_cosine_segments   <- normalization + peak detection + saddle-point splitting
  - slice_physical_mp4     <- unchanged helper (re-encodes one sub-clip)
  - slice_video_segments   <- per-video subfolder creation + slicing + Excel row building
  - save_segment_excel     <- merge/de-dupe/save into the master workbook
"""

import cv2
import numpy as np
import os
import scipy.spatial.distance as dist
from scipy.signal import find_peaks, savgol_filter
import pandas as pd


# ==============================================================================
# ── compute_histogram ──────────────────────────────────────────────────────────
# ==============================================================================
def compute_histogram(frame, bins=64):
    """Convert a BGR frame to grayscale and return its normalized intensity histogram."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
    hist = hist / (np.sum(hist) + 1e-9)  # Normalize so the histogram sums to 1 (avoid divide-by-zero)
    return hist.flatten()


# ==============================================================================
# ── extract_cosine_curve ──────────────────────────────────────────────────────
# ==============================================================================
def extract_cosine_curve(video_path, savgol_window=7, savgol_poly=3, bins=64):
    """
    Opens a video, samples exactly 1 frame/sec, and computes the Savitzky-Golay
    smoothed Cosine histogram-distance curve between consecutive frames.

    Returns (timestamps, fps, cosine_smoothed), or None if the video is
    corrupt/unopenable or doesn't have enough frames to compare.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Skip videos that fail to open or report a 0 FPS (corrupt/unreadable files)
    if not cap.isOpened() or fps == 0:
        cap.release()
        return None

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
        return None  # Not enough frames to compute frame-to-frame change

    # Build per-second histograms, then measure cosine distance between consecutive frames
    histograms_norm = [compute_histogram(f, bins) for f in frames]
    cosine_raw = np.zeros(n_frames)
    for i in range(1, n_frames):
        cosine_raw[i] = dist.cosine(histograms_norm[i - 1], histograms_norm[i])

    # Smooth the raw change curve with Savitzky-Golay to suppress frame-flicker noise
    if len(cosine_raw) > savgol_window:
        cosine_smoothed = savgol_filter(cosine_raw, window_length=savgol_window, polyorder=savgol_poly)
    else:
        cosine_smoothed = np.copy(cosine_raw)

    return np.array(timestamps), fps, cosine_smoothed


# ==============================================================================
# ── find_cosine_segments ──────────────────────────────────────────────────────
# ==============================================================================
def find_cosine_segments(cosine_smoothed, noise_floor, peak_prominence_factor,
                          min_peak_height, min_peak_distance_secs, base_slope_cutoff):
    """
    Normalizes the cosine curve to 0-1, runs SciPy's peak detector with prominence/
    height/distance constraints, recovers peaks sitting right at the video's edges,
    then walks each peak down to its saddle-point valleys to find the true start/end.

    Returns a list of segment dicts: {'start_idx', 'peak_idx', 'end_idx'}.
    """
    # Normalize the smoothed curve to a 0-1 scale, with a noise floor safety guard
    local_max = np.max(cosine_smoothed)
    divisor = max(local_max, noise_floor)
    scaled_smoothed = cosine_smoothed / divisor
    curve_range = np.max(scaled_smoothed) - np.min(scaled_smoothed)

    # Run SciPy's peak detector with prominence/height/distance constraints tuned above
    detected_peaks, _ = find_peaks(
        scaled_smoothed,
        prominence=peak_prominence_factor * curve_range,
        height=min_peak_height,
        distance=min_peak_distance_secs
    )
    peaks = sorted(list(set(detected_peaks)))

    # ── BOUNDARY EDGE SPIKE RECOVERY ──────────────────────────────────────────
    # Catches activity that's already happening right at the very start or end
    # of the video, which find_peaks can miss since it can't see past the edges.
    edge_frame_window = int(min_peak_distance_secs)

    if len(scaled_smoothed) > edge_frame_window:
        front_max_idx = np.argmax(scaled_smoothed[:edge_frame_window])
        if scaled_smoothed[front_max_idx] >= min_peak_height:
            if not any(abs(front_max_idx - p) < min_peak_distance_secs for p in peaks):
                peaks.insert(0, front_max_idx)

    if len(scaled_smoothed) > edge_frame_window:
        back_start_idx = len(scaled_smoothed) - edge_frame_window
        back_max_idx = back_start_idx + np.argmax(scaled_smoothed[back_start_idx:])
        if scaled_smoothed[back_max_idx] >= min_peak_height:
            if not any(abs(back_max_idx - p) < min_peak_distance_secs for p in peaks):
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
        thresh_l = v_left_floor + base_slope_cutoff * (v_peak - v_left_floor)
        thresh_r = v_right_floor + base_slope_cutoff * (v_peak - v_right_floor)

        start_idx = p_idx
        while start_idx > left_valley_idx and scaled_smoothed[start_idx] > thresh_l:
            start_idx -= 1

        end_idx = p_idx
        while end_idx < right_valley_idx and scaled_smoothed[end_idx] > thresh_r:
            end_idx += 1

        segments.append({'start_idx': start_idx, 'peak_idx': p_idx, 'end_idx': end_idx})

    return segments


# ==============================================================================
# ── slice_physical_mp4 ────────────────────────────────────────────────────────
# ==============================================================================
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
# ── slice_video_segments ──────────────────────────────────────────────────────
# ==============================================================================
def slice_video_segments(v_path, v_name, segments, timestamps, fps, segment_dir):
    """
    Makes a per-video subfolder under segment_dir, physically slices each segment's
    .mp4 clip into it via slice_physical_mp4, and returns the Excel row dicts to log.
    """
    # Make a per-video subfolder to hold its sliced segment clips
    video_base_name = os.path.splitext(v_name)[0]
    video_output_dir = os.path.join(segment_dir, video_base_name)
    os.makedirs(video_output_dir, exist_ok=True)

    excel_rows = []
    for seg_idx, seg in enumerate(segments, 1):
        t_start = timestamps[seg['start_idx']]
        t_end = timestamps[seg['end_idx']]

        segment_clip_name = f"segment_{seg_idx}.mp4"
        segment_clip_path = os.path.join(video_output_dir, segment_clip_name)

        print(f"    ✂️ Slicing -> {segment_clip_name} ({t_start:.1f}s to {t_end:.1f}s)...")
        slice_physical_mp4(v_path, t_start, t_end, segment_clip_path, fps)

        # 🌟 CRITICAL: Saving the data to the memory list for Excel
        excel_rows.append({
            "Source_Video": v_name,
            "Segment_ID": seg_idx,
            "Start_Time": round(t_start, 2),
            "End_Time": round(t_end, 2)
        })

    return excel_rows


# ==============================================================================
# ── save_segment_excel ────────────────────────────────────────────────────────
# ==============================================================================
def save_segment_excel(excel_metadata_list, excel_path):
    """
    Merges today's fresh segment rows into any existing workbook, de-duplicating
    re-scanned segments by (Source_Video, Segment_ID, Start_Time), and saves it.
    """
    if len(excel_metadata_list) == 0:
        print("\n⚠️ No movement detected across any videos. Excel sheet was not created.")
        return

    df_new = pd.DataFrame(excel_metadata_list)

    # Merge with any pre-existing workbook, de-duplicating re-scanned segments
    if os.path.exists(excel_path):
        try:
            df_historical = pd.read_excel(excel_path)
            df_combined = pd.concat([df_historical, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=["Source_Video", "Segment_ID", "Start_Time"], keep="last")
        except Exception:
            df_combined = df_new
    else:
        df_combined = df_new

    df_combined.to_excel(excel_path, index=False)
    print(f"\n✅ SUCCESS! Master timeline saved to:\n➡️ {excel_path}")