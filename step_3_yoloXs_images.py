# -*- coding: utf-8 -*-
"""
Gemma_YOLOXs_images.py

We pass the VIDEO SEGMENTS into the YOLOX-s model so that we can detect the
TRUCKS and then ASSIGN ID's to them, get their IN & OUT TIME, the NUMBER OF
WHEELS, TRUCK IMAGES, PLATE IMAGES and SAVE them to an EXCEL SHEET (with
clickable links to the local image files) and the TRUCK AND PLATE IMAGES are
also saved in a local folder, and we are also saving the YOLOX-s detected
VIDEO SEGMENTS into a local folder.

REFACTOR NOTE: The long processing chunks (ONNX session loading, per-segment
detect+track loop, track merging, database compilation, Excel export) now
live in yolox_pipeline_lib.py as callable functions/classes. This file just
holds your config/tuning parameters and orchestrates the per-segment loop by
calling into that library.

The whole thing is wrapped in main() so this file can either be run directly
(`python Gemma_YOLOXs_images.py`) or imported and called from a higher-level
orchestrator script, e.g.:
    import gemma_yoloxs_images
    gemma_yoloxs_images.main()

NOTE: Before running, in your terminal:
    pip install paddlepaddle paddleocr onnxruntime pandas openpyxl opencv-python numpy

The YOLOX-S ONNX weights are auto-downloaded into ONNX_PATH on first run if
missing — no manual `!wget` needed.

NOTE: Fully local — no Google Drive, no Drive API, no OAuth. Excel hyperlinks
point straight at local .jpg paths — click "View Truck"/"View Plate" to open.

Make sure yolox_pipeline_lib.py is in the same folder as this script (or
somewhere on your PYTHONPATH) so the import below resolves.
"""

import os
import shutil
import pandas as pd

from step_3_functions_yoloXs_images import (
    ensure_yolox_weights,
    load_yolox_session,
    run_detection_tracking,
    merge_broken_tracks,
    compile_segment_database,
    save_vehicle_registry,
)


def main():
    # ==============================================================================
    # ── SECTION 1: MASTER TUNING PARAMETERS & PATHS CONFIGURATION ─────────────────
    # ==============================================================================

    # 💾 1A: ENVIRONMENT DIRECTORIES & TRACKING PATHS
    # ------------------------------------------------------------------------------
    # 🔧 EDIT THESE to point at your local equivalents of the Drive folders
    # (e.g. your local Google Drive Desktop sync mirror, or any plain local folder).
    TIMESTAMPS_EXCEL = r"D:\Traffic_Control\segment_timestamps.xlsx"
    SEGMENT_DIR      = r"D:\Traffic_Control\trial_video_segments"
    ASSET_DIR        = r"D:\Traffic_Control\final_assets"
    ONNX_PATH        = r"D:\Traffic_Control\yolox_small.onnx"
    TEMP_DIR         = r"D:\Traffic_Control\temp"

    # 🧠 1B: DETECTOR INPUT CORE SETTINGS
    # ------------------------------------------------------------------------------
    INPUT_SIZE       = (640, 640)  # Dimensions to compress frames for the ONNX grid.
    TRUCK_CLASS_ID   = 7           # COCO dataset index for trucks. DO NOT CHANGE.

    # 🎯 1C: DETECTOR CONFIDENCE HYPER-PARAMETERS
    # ------------------------------------------------------------------------------
    DETECTION_SCORE_THR = 0.45
    """
    What it does: Minimum confidence score required for the AI to recognize a truck.
    If you INCREASE this value (e.g., to 0.50):
      - Pro: Guarantees 100% pure trucks; completely avoids misidentifying large SUVs.
      - Con: The AI will fail to detect trucks in deep shadows, heavy rain, or night conditions.
    If you DECREASE this value (e.g., to 0.25):
      - Pro: Highly aggressive; catches faint, blurry, or distant trucks early at the frame boundary.
      - Con: Risks letting large passenger vans or buses leak into your database log.
    """

    DETECTION_NMS_THR   = 0.6
    """
    What it does: Non-Maximum Suppression threshold to smash overlapping bounding boxes.
    If you INCREASE this value (e.g., to 0.60):
      - Allows more overlapping boxes to live. Use only if multiple trucks cross the screen tightly side-by-side.
    If you DECREASE this value (e.g., to 0.30):
      - Highly restrictive. Smashes multiple boxes together into a single master box.
    """

    # 👣 1D: LIVE INTRA-FRAME LIGHTWEIGHT TRACKER PATIENCE
    # ------------------------------------------------------------------------------
    TRACKER_MAX_DISAPPEARED = 60
    """
    What it does: The number of consecutive frames the tracker waits before giving up on an ID.
    If you INCREASE this value (e.g., to 60):
      - Pro: Keeps the vehicle ID perfectly locked even if it passes behind moderately thick signs or light poles.
      - Con: If a truck exits and a completely different truck enters from the same lane immediately, it might inherit the old ID.
    If you DECREASE this value (e.g., to 15):
      - Pro: Highly precise local frame assignments; prevents ID inheritance swaps in fast heavy traffic lanes.
      - Con: Causes a clean track to split immediately into new duplicate rows if a truck passes behind a single tree branch.
    """

    TRACKER_DISTANCE_THR    = 200
    """
    What it does: Maximum pixel distance a vehicle's center can travel between frames to remain linked.
    If you INCREASE this value (e.g., to 250):
      - Necessary for high-speed expressways where a truck travels massive screen distances frame-to-frame.
    If you DECREASE this value (e.g., to 100):
      - Best for slow-moving, congested junctions. Prevents track vectors from jumping between adjacent trucks.
    """

    # 🌳 1E: POST-VIDEO SPATIAL-TEMPORAL MERGE ENGINE CONSTANTS
    # ------------------------------------------------------------------------------
    MERGE_MAX_TIME_GAP   = 2.5
    """
    What it does: The maximum duration window (in seconds) allowed between a track ending and a new one starting to trigger a merge.
    If you INCREASE this value (e.g., to 3.5):
      - Pro: Safely bridges massive visual blockages, such as a truck disappearing entirely behind a wide building pillar or thick tree clump.
      - Con: If trucks pass sequentially in identical lanes with a high frame rate, separate vehicles can get combined into a single ledger row.
    If you DECREASE this value (e.g., to 0.8):
      - Enforces near-perfect temporal continuity; zero risk of mixing up sequential vehicles, but splits tracks if a vehicle stops completely.
    """

    MERGE_MAX_SPATIAL_GAP = 350
    """
    What it does: Maximum pixel translation gap between where Track A vanished and Track B initialized.
    If you INCREASE this value (e.g., to 500):
      - Accounts for rapid swerving or camera panning, but risks merging separate vehicles across separate lanes.
    If you DECREASE this value (e.g., to 150):
      - Restricts mergers strictly to trucks maintaining a razor-straight lane trajectory vector.
    """

    # 🛡️ 1F: PRODUCTION DATA SYSTEM CLEANERS
    # ------------------------------------------------------------------------------
    MIN_VALID_FRAMES_LOGGED = 4
    """
    What it does: Hard verification filter. Minimum cumulative frames tracked before committing to the Excel registry.
    If you INCREASE this value (e.g., to 15):
      - 100% foolproof defense against transient environmental noise. Completely filters out temporary cloud shadow movements.
      - Con: Drops data for trucks traveling at extreme high speeds that cross the screen boundary in under half a second.
    If you DECREASE this value (e.g., to 2):
      - Logs everything, including brief edge-passing glimpses, but risks tracking false AI pixel flickers as real vehicles.
    """

    COLLISION_STD_GUARD     = 30.0
    """
    What it does: Bounding box standard deviation variance watcher across a rolling 3-frame average.
    If the width of a box spikes higher than this number, it flags a "Collision/Overlap Zone".
      - Lowering it (e.g., 15.0) makes the system more sensitive, freezing aspect-ratio checks the moment trucks pass closely.
    """

    # ✂️ 1G: GEOMETRIC HEURISTIC PLATE CROP PROPORTIONS
    # ------------------------------------------------------------------------------
    PLATE_MARGIN_WIDTH_CLIP  = 0.20  # Crops out 20% from the left and right margins of the box to isolate the center.
    PLATE_BOTTOM_HEIGHT_CLIP = 0.35  # Extracts strictly the bottom 35% height profile of the chassis where plates are mounted.

    # ==============================================================================
    # ── SYSTEM OVERHEAD SETUP ─────────────────────────────────────────────────────
    # ==============================================================================

    # Initialize target output sub-directories for structured storage safely
    os.makedirs(ASSET_DIR, exist_ok=True)
    os.makedirs(os.path.join(ASSET_DIR, "truck_crops"), exist_ok=True)
    os.makedirs(os.path.join(ASSET_DIR, "plate_crops"), exist_ok=True)
    os.makedirs(os.path.join(ASSET_DIR, "annotated_segments"), exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    YOLOX_WEIGHTS_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx"
    ensure_yolox_weights(ONNX_PATH, YOLOX_WEIGHTS_URL)

    # ==============================================================================
    # ── SECTION 2: LOAD AI ENGINES INTO CPU ───────────────────────────────────────
    # ==============================================================================
    print("🚀 Loading YOLOX-S (Apache 2.0)...")
    ort_session, input_name = load_yolox_session(ONNX_PATH)

    # ==============================================================================
    # ── SECTION 4: MASTER TIMELINE EXECUTION LOOP ─────────────────────────────────
    # ==============================================================================

    # Load sheet containing macro temporal cuts generated during the video segmentation layer
    df_times = pd.read_excel(TIMESTAMPS_EXCEL)
    final_database = []

    print(f"\n📦 Loaded Metadata for {len(df_times)} segments. Beginning precision extraction...")

    # Iterate through every temporal video slice segment sequentially
    for idx, row in df_times.iterrows():
        master_video_name = row["Source_Video"]
        seg_id = row["Segment_ID"]
        t_start = row["Start_Time"]

        segment_filename = f"segment_{seg_id}.mp4"
        video_base_name = os.path.splitext(master_video_name)[0]
        drive_video_path = os.path.join(SEGMENT_DIR, video_base_name, segment_filename)

        if not os.path.exists(drive_video_path):
            continue

        print(f"\n🎬 Processing Segment {seg_id} (Absolute Anchor: {t_start}s)...")

        # Isolate I/O bottlenecks by copying network files to the local scratch workspace
        local_video_path = os.path.join(TEMP_DIR, f"temp_in_{segment_filename}")
        local_out_vid = os.path.join(TEMP_DIR, f"temp_out_{segment_filename}")
        shutil.copy(drive_video_path, local_video_path)

        active_state_buffer = run_detection_tracking(
            local_video_path, local_out_vid, t_start,
            ort_session, input_name, INPUT_SIZE, TRUCK_CLASS_ID,
            DETECTION_SCORE_THR, DETECTION_NMS_THR,
            TRACKER_MAX_DISAPPEARED, TRACKER_DISTANCE_THR,
            COLLISION_STD_GUARD, PLATE_MARGIN_WIDTH_CLIP, PLATE_BOTTOM_HEIGHT_CLIP
        )

        # Save finalized annotated video file back to the primary asset folder
        final_video_dest = os.path.join(ASSET_DIR, "annotated_segments", f"annotated_{segment_filename}")
        shutil.copy(local_out_vid, final_video_dest)

        # Purge intermediate storage allocations to prevent local workspace bloating
        if os.path.exists(local_video_path): os.remove(local_video_path)
        if os.path.exists(local_out_vid): os.remove(local_out_vid)

        active_state_buffer = merge_broken_tracks(active_state_buffer, MERGE_MAX_TIME_GAP, MERGE_MAX_SPATIAL_GAP)

        segment_rows = compile_segment_database(
            active_state_buffer, seg_id, master_video_name, ASSET_DIR, MIN_VALID_FRAMES_LOGGED
        )
        final_database.extend(segment_rows)

    # ==============================================================================
    # ── SECTION 5: EXPORT CLEAN EXCEL TABLE ───────────────────────────────────────
    # ==============================================================================
    save_vehicle_registry(final_database, ASSET_DIR)


if __name__ == "__main__":
    main()