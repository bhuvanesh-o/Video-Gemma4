# -*- coding: utf-8 -*-
"""
step_3_yoloXs_images.py

We pass the VIDEO SEGMENTS into the YOLOX-s model so that we can detect the
TRUCKS and then ASSIGN ID's to them, get their IN & OUT TIME, the NUMBER OF
WHEELS, TRUCK IMAGES, PLATE IMAGES and SAVE them to an EXCEL SHEET (with
clickable links to the local image files) and the TRUCK AND PLATE IMAGES are
also saved in a local folder, and we are also saving the YOLOX-s detected
VIDEO SEGMENTS into a local folder.

REFACTOR NOTE: The long processing chunks (ONNX session loading, per-segment
detect+track loop, track merging, database compilation, Excel export) now
live in step_3_functions_yoloXs_images.py as callable functions/classes. This file just
holds your config/tuning parameters and orchestrates the per-segment loop by
calling into that library.

The whole thing is wrapped in main() so this file can either be run directly
(`python step_3_yoloXs_images.py`) or imported and called from a higher-level
orchestrator script, e.g.:
    import step_3_yoloXs_images
    step_3_yoloXs_images.main(job_id="local_test")

REFACTOR NOTE 2: Hardcoded D:\\Traffic_Control paths are replaced by storage.path_for()
so every run is isolated to its own job folder. print() is replaced by emit() from
progress.py so progress can be streamed to a browser later via FastAPI SSE while
still printing locally when run from the command line.

REFACTOR NOTE 3 (structured events): every emit() call below now also passes
stage="detection" and a short ui_message for the frontend's narrative UI.

NOTE: Before running, in your terminal:
    pip install paddlepaddle paddleocr onnxruntime pandas openpyxl opencv-python numpy

The YOLOX-S ONNX weights are auto-downloaded into ONNX_PATH on first run if
missing — no manual `!wget` needed.

NOTE: Fully local — no Google Drive, no Drive API, no OAuth. Excel hyperlinks
point straight at local .jpg paths — click "View Truck"/"View Plate" to open.

Make sure step_3_functions_yoloXs_images.py, storage.py, and progress.py are in the
same folder as this script (or somewhere on your PYTHONPATH) so the imports below resolve.
"""
import os
import shutil
import pandas as pd
import cv2
import subprocess

import openvino.properties as props


from step_3_functions_yoloXs_images import (
    ensure_yolox_weights,
    load_yolox_session,
    run_detection_tracking,
    merge_broken_tracks,
    compile_segment_database,
    save_vehicle_registry,
)
from storage import path_for, dir_for
from progress import emit


def main(job_id, model_precision = None, performance_hint=None):
    # ==============================================================================
    # ── SECTION 1: MASTER TUNING PARAMETERS & PATHS CONFIGURATION ─────────────────
    # ==============================================================================
    # 💾 1A: ENVIRONMENT DIRECTORIES & TRACKING PATHS
    # ------------------------------------------------------------------------------
    # Replaces hardcoded D:\Traffic_Control paths — same job folder step 1/2 wrote to.
    TIMESTAMPS_EXCEL = path_for(job_id, "segment_timestamps.xlsx")
    SEGMENT_DIR      = dir_for(job_id, "trial_video_segments")
    ASSET_DIR        = dir_for(job_id, "final_assets")
    ONNX_PATH        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "yolox_small.onnx")
    INT8_PATH        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "yolox_small_int8.xml")
    TEMP_DIR         = dir_for(job_id, "temp")

    # 🧠 1B: DETECTOR INPUT CORE SETTINGS
    # ------------------------------------------------------------------------------
    INPUT_SIZE       = (640, 640)  # Dimensions to compress frames for the ONNX grid.
    TRUCK_CLASS_ID   = 7           # COCO dataset index for trucks. DO NOT CHANGE.

    # ⚙️ MODEL PRECISION SWITCH — toggle between FP32 (baseline) and INT8 (faster,
    # small accuracy tradeoff). INT8 requires weights/yolox_small_int8.xml to
    # already exist — run build_calibration_data.py + quantize_int8.py first.

    MODEL_PRECISION = "INT8"   # "FP32" or "INT8"

    MODEL_PATH = INT8_PATH if MODEL_PRECISION == "INT8" else ONNX_PATH
    if MODEL_PRECISION == "INT8" and not os.path.exists(INT8_PATH):
        raise FileNotFoundError(
            f"{INT8_PATH} not found. Run build_calibration_data.py then quantize_int8.py "
            f"first, or set MODEL_PRECISION back to 'FP32'."
        )

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

    # 🖼️ 1H: BEST-CROP SELECTION (truck photo + plate crop quality)
    # ------------------------------------------------------------------------------
    MAX_CANDIDATE_FRAMES = 5
    PLATE_CONFIDENCE_THRESHOLD = 0.5
    EARLY_EXIT_CONFIDENCE = 0.85
    COLOR_SCORE_WEIGHT = 0.4          # how much plate-color-likelihood counts vs raw detection confidence
    RECOGNITION_CHECK_TOP_N = 2       # only these many top regions get the (more expensive) recognition tiebreak
    """
    Once a candidate's detected plate confidence reaches this, stop checking
    further candidates for that truck — the current best is trusted as good
    enough. Lower this to exit earlier (faster, more risk of settling for a
    mediocre plate when a better one was in a later candidate); raise it to
    be more thorough (slower, but only settles for very confident detections).
    """

    
        # 📐 1I: APPROACH-PHASE + SIZE SWEET-SPOT FILTERING
    # ------------------------------------------------------------------------------
    SIZE_SWEET_SPOT_MIN = 0.15   # truck's box must occupy at least 10% of frame area to be a candidate
    SIZE_SWEET_SPOT_MAX = 0.70   # ...and no more than 25% — beyond this, likely too close/distorted
    """
    Placeholder starting values — calibrate against real footage: pause a
    video where the truck's front is clearly angled toward your camera (not
    yet side-on) and check what fraction of frame area its box occupies at
    that moment. Adjust these two numbers to match what you actually see.
    """
    RECEDE_TOLERANCE = 0.90
    """
    Once box width drops below (peak_width_so_far * RECEDE_TOLERANCE), the
    truck is considered past its closest point and receding — no further
    candidate frames are collected for it after that. Lower (e.g. 0.80) =
    more tolerant of width jitter before declaring "receding"; higher (e.g.
    0.95) = declares receding sooner, more conservative about approach-only.
    """
    MIN_CANDIDATE_SPACING_SECONDS = 0.5
    """
    Minimum time gap enforced between any two frames in a truck's shortlist —
    prevents 5 near-duplicate frames from one lucky stretch dominating the
    shortlist. Expressed in seconds (not frames) so it behaves consistently
    across videos with different frame rates.
    """

    

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
    ensure_yolox_weights(ONNX_PATH, YOLOX_WEIGHTS_URL, job_id=job_id)

    # ==============================================================================
    # ── SECTION 2: LOAD AI ENGINES INTO CPU ───────────────────────────────────────
    # ==============================================================================
    emit(f"🚀 Loading YOLOX-S via OpenVINO ({MODEL_PRECISION}, CPU)...", job_id=job_id,
         stage="detection", ui_message="Loading the truck detection model...")
    OPENVINO_DEVICE = "CPU"
    PERFORMANCE_HINT = performance_hint or "THROUGHPUT"   # 👈 THIS is the switch — change "LATENCY" to "THROUGHPUT" here, or pass it in from the caller


    def check_optimal_requests(compiled_model, job_id=None):
            num = compiled_model.get_property(props.optimal_number_of_infer_requests)
            emit(f"⚙️ OpenVINO recommends {num} infer requests for this model/device/hint combo.",
                job_id=job_id, stage="detection", ui_message="")
            return num

    
    compiled_model, input_layer, output_layer = load_yolox_session(MODEL_PATH, device=OPENVINO_DEVICE, performance_hint=PERFORMANCE_HINT, job_id=job_id)


    check_optimal_requests(compiled_model, job_id=job_id)  # run once, note the number, can remove after

    # Two infer requests, created FRESH for this job (never cached, never
    # shared across jobs) — this is what #11's isolation requirement means
    # concretely. Only compiled_model above is shared, via get_compiled_model's
    # lru_cache from #6.

    '''
    infer_request_a = compiled_model.create_infer_request()
    infer_request_b = compiled_model.create_infer_request()
    '''


    NUM_INFER_REQUESTS = 4

    infer_requests = [
        compiled_model.create_infer_request()
        for _ in range(NUM_INFER_REQUESTS)
    ]




    # Add this right after get_compiled_model() returns, anywhere convenient
    # for a one-off check — e.g. temporarily inside load_yolox_session(), or
    # as a tiny standalone script.

    




    # ==============================================================================
    # ── SECTION 4: MASTER TIMELINE EXECUTION LOOP ─────────────────────────────────
    # ==============================================================================
    # Load sheet containing macro temporal cuts generated during the video segmentation layer

    if not os.path.exists(TIMESTAMPS_EXCEL):
        emit(f"⚠️ No segment timestamps file found — likely no motion/trucks detected upstream. Skipping detection.",
             job_id=job_id, stage="detection", ui_message="No activity detected in this footage — nothing to scan for trucks.")
        save_vehicle_registry([], ASSET_DIR, job_id=job_id)  # writes the empty headers-only registry, so /download doesn't 404
        return
    
    df_times = pd.read_excel(TIMESTAMPS_EXCEL)
    final_database = []
    emit(f"\n📦 Loaded Metadata for {len(df_times)} segments. Beginning precision extraction...", job_id=job_id,
         stage="detection", ui_message=f"Scanning {len(df_times)} clips for trucks...")

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

        emit(f"\n🎬 Processing Segment {seg_id} (Absolute Anchor: {t_start}s)...", job_id=job_id,
             stage="detection", ui_message=f"Looking for trucks in clip {seg_id}...")

        # Isolate I/O bottlenecks by copying network files to the local scratch workspace
        local_video_path = os.path.join(TEMP_DIR, f"temp_in_{segment_filename}")
        local_out_vid = os.path.join(TEMP_DIR, f"temp_out_{segment_filename}")
        shutil.copy(drive_video_path, local_video_path)

        # inside the per-segment "for idx, row in df_times.iterrows():" loop, replace the
        # run_detection_tracking(...) call with:

        active_state_buffer, segment_timing = run_detection_tracking(
          local_video_path, local_out_vid, t_start,
          infer_requests, input_layer, output_layer, INPUT_SIZE, TRUCK_CLASS_ID,
          DETECTION_SCORE_THR, DETECTION_NMS_THR,
          TRACKER_MAX_DISAPPEARED, TRACKER_DISTANCE_THR,
          COLLISION_STD_GUARD, MAX_CANDIDATE_FRAMES,
          SIZE_SWEET_SPOT_MIN, SIZE_SWEET_SPOT_MAX, RECEDE_TOLERANCE, MIN_CANDIDATE_SPACING_SECONDS,
          job_id=job_id
        )

        
        # Save finalized annotated video file back to the primary asset folder
        '''
        final_video_dest = os.path.join(ASSET_DIR, "annotated_segments", f"annotated_{segment_filename}")
        shutil.copy(local_out_vid, final_video_dest)
        '''

        final_video_dest = os.path.join(ASSET_DIR, "annotated_segments", f"annotated_{segment_filename}")
        # mp4v (OpenCV's default fourcc) isn't decodable by browsers — re-encode to H.264 so <video> can play it
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", local_out_vid, "-vcodec", "libx264", "-pix_fmt", "yuv420p", final_video_dest],
                check=True, capture_output=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            emit(f"⚠️ ffmpeg re-encode failed ({e}) — copying raw file (may not play in browser).",
                job_id=job_id, stage="detection", ui_message="")
            shutil.copy(local_out_vid, final_video_dest)

        # Purge intermediate storage allocations to prevent local workspace bloating
        if os.path.exists(local_video_path): os.remove(local_video_path)
        if os.path.exists(local_out_vid): os.remove(local_out_vid)


        # fps needed for merge's spacing dedup — read it once from the segment file
        _cap_for_fps = cv2.VideoCapture(drive_video_path)
        segment_fps = _cap_for_fps.get(cv2.CAP_PROP_FPS) or 30.0
        _cap_for_fps.release()

        active_state_buffer = merge_broken_tracks(
            active_state_buffer, MERGE_MAX_TIME_GAP, MERGE_MAX_SPATIAL_GAP,
            MAX_CANDIDATE_FRAMES, MIN_CANDIDATE_SPACING_SECONDS, segment_fps
        )


        segment_rows = compile_segment_database(
            active_state_buffer, seg_id, master_video_name, ASSET_DIR, MIN_VALID_FRAMES_LOGGED,
            PLATE_MARGIN_WIDTH_CLIP, PLATE_BOTTOM_HEIGHT_CLIP,
            plate_confidence_thr=PLATE_CONFIDENCE_THRESHOLD,
            early_exit_confidence=EARLY_EXIT_CONFIDENCE,
            color_score_weight=COLOR_SCORE_WEIGHT,
            recognition_check_top_n=RECOGNITION_CHECK_TOP_N,
            job_id=job_id
        )

        final_database.extend(segment_rows)

    # ==============================================================================
    # ── SECTION 5: EXPORT CLEAN EXCEL TABLE ───────────────────────────────────────
    # ==============================================================================
    save_vehicle_registry(final_database, ASSET_DIR, job_id=job_id)


if __name__ == "__main__":
    # Local manual test: run step_1 and step_2 first (or manually populate the
    # job folder equivalents), then run this file directly.
    main(job_id="local_test")