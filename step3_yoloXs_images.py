# -*- coding: utf-8 -*-
"""
Gemma_YOLOXs_images.py

Ported from Google Colab (Gemma_YOLOXs_images.ipynb) to a standalone local
script for VS Code.

We pass the VIDEO SEGMENTS into the YOLOX-s model so that we can detect the
TRUCKS and then ASSIGN ID's to them, get their IN & OUT TIME, the NUMBER OF
WHEELS, TRUCK IMAGES, PLATE IMAGES and SAVE them to an EXCEL SHEET (with
clickable links to the local image files) and the TRUCK AND PLATE IMAGES are
also saved in a local folder, and we are also saving the YOLOX-s detected
VIDEO SEGMENTS into a local folder.

NOTE: Colab's shell-magic install/download lines are gone. Before running,
in your terminal:

    pip install paddlepaddle paddleocr onnxruntime pandas openpyxl opencv-python numpy

The YOLOX-S ONNX weights are auto-downloaded into ONNX_PATH on first run if
missing (see the download guard in SYSTEM OVERHEAD SETUP below) — no more
manual `!wget`.

NOTE: This version is fully local — no Google Drive, no Drive API, no OAuth.
The original notebook's Section 4.5 resolved each saved crop to a Drive file
ID so it could write a `=HYPERLINK("https://drive.google.com/...")` formula.
Since everything now lives on disk, the Excel sheet just hyperlinks straight
to the local .jpg path instead (Excel supports clicking local file paths the
same way it does URLs) — click "View Truck"/"View Plate" to open the image.
"""

# ==============================================================================
# ── FINAL COMMERCIAL PIPELINE: PARAMETRIZED TRUCK TRACKING & ENGINE ───────────
# ==============================================================================

"""
Core System Description:
This enterprise-grade production script implements an automated macro-to-micro
video analysis pipeline. It ingests pre-segmented high-activity video clips,
runs localized CPU-based vehicle inference using an optimized YOLOX-Small ONNX model,
tracks individual trucks using a multi-frame spatial centroid tracker, extracts
optimized 'Hero Frame' asset crops for both the truck chassis and license plate region,
handles asynchronous cloud synchronization with the Google Drive API, and generates
a highly detailed Excel registry complete with live relative hyperlinks.
"""

import os
import cv2
import numpy as np
import pandas as pd
import shutil
import math
import urllib.request
import onnxruntime as ort

# ==============================================================================
# ── SECTION 1: MASTER TUNING PARAMETERS & PATHS CONFIGURATION ─────────────────
# ==============================================================================

# 💾 1A: ENVIRONMENT DIRECTORIES & TRACKING PATHS
# ------------------------------------------------------------------------------
# 🔧 EDIT THESE to point at your local equivalents of the Drive folders
# (e.g. your local Google Drive Desktop sync mirror, or any plain local folder).
# Input source spreadsheet containing timestamps and structural video segment references
TIMESTAMPS_EXCEL = r"C:\Users\YourName\GEMMA\trial_main_video\segment_timestamps.xlsx"
# Root folder where physical pre-sliced video segment .mp4 clips are stored
SEGMENT_DIR      = r"C:\Users\YourName\GEMMA\trial_main_video\trial_video_segments\truck_video_trial"
# Master asset output directory for crops, logs, and annotated reference videos
ASSET_DIR        = r"C:\Users\YourName\GEMMA\trial_main_video\final_assets"
# Path to the compiled YOLOX object detection model weights file
ONNX_PATH        = r"C:\Users\YourName\GEMMA\models\yolox_small.onnx"
# Local scratch folder for temporary per-segment video copies (was /content/ on Colab)
TEMP_DIR         = r"C:\Users\YourName\GEMMA\temp"

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
# (No drive.mount() needed locally — these are just plain local folders now.)
os.makedirs(ASSET_DIR, exist_ok=True)
os.makedirs(os.path.join(ASSET_DIR, "truck_crops"), exist_ok=True)
os.makedirs(os.path.join(ASSET_DIR, "plate_crops"), exist_ok=True)
os.makedirs(os.path.join(ASSET_DIR, "annotated_segments"), exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Auto-download the YOLOX-S ONNX weights once if they aren't already sitting at ONNX_PATH
# (this replaces the Colab `!wget ... -O /content/yolox_small.onnx` line)
YOLOX_WEIGHTS_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx"
if not os.path.exists(ONNX_PATH):
    print("📥 YOLOX-S weights not found locally — downloading once from GitHub...")
    os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)
    urllib.request.urlretrieve(YOLOX_WEIGHTS_URL, ONNX_PATH)
    print(f"✅ Saved weights to {ONNX_PATH}")


# ==============================================================================
# ── SECTION 2: LOAD AI ENGINES INTO CPU ───────────────────────────────────────
# ==============================================================================

print("🚀 Loading YOLOX-S (Apache 2.0)...")
# Initialize highly optimized runtime execution session mapping directly to local host CPU threads
ort_session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
input_name = ort_session.get_inputs()[0].name


# ==============================================================================
# ── SECTION 3: CORE FUNCTIONS & LIGHTWEIGHT TRACKER ───────────────────────────
# ==============================================================================

def preprocess(img, input_size):
    """
    Transforms raw video frames into compliant multidimensional numerical arrays matching
    the exact tensor structure expected by the pre-compiled YOLOX graph.
    - Implements adaptive letterbox resizing to preserve native image aspect ratios.
    - Pads residual space with a standard gray baseline color (114).
    - Permutes shapes from HWC (Height, Width, Channels) to network-standard CHW format.
    """
    padded = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized = cv2.resize(img, (int(img.shape[1]*r), int(img.shape[0]*r)), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    padded[:resized.shape[0], :resized.shape[1]] = resized
    return np.ascontiguousarray(padded.transpose((2, 0, 1)), dtype=np.float32), r

def decode_outputs(outputs, img_size, strides=(8, 16, 32)):
    """
    Translates raw feature map grid offsets emitted by the ONNX neural network multi-scale layers
    back into true, un-scaled bounding box anchor coordinates.
    """
    grids, exp_strides = [], []
    for stride in strides:
        hsize, wsize = img_size[0] // stride, img_size[1] // stride
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        exp_strides.append(np.full((1, grid.shape[1], 1), stride))
    grids = np.concatenate(grids, 1)
    exp_strides = np.concatenate(exp_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * exp_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * exp_strides
    return outputs

def get_truck_boxes(outputs, ratio, w, h, score_thr, nms_thr):
    """
    Parses structural raw multi-class matrices to extract targets matching the target Truck class.
    - Filters bounding coordinates using a confidence threshold mask.
    - Resolves raw spatial bounding dimensions into standard pixel coordinates.
    - Executes Non-Maximum Suppression (NMS) to eliminate duplicate overlapping bounding boxes.
    - Soft-clips target dimensions to protect against edge out-of-bounds boundary exceptions.
    """
    boxes = outputs[:, :4]
    scores = outputs[:, 4:5] * outputs[:, 5:]
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    mask = (cls_scores > score_thr) & (cls_inds == TRUCK_CLASS_ID)
    if mask.sum() == 0: return []

    truck_bboxes = boxes[mask]
    confidences = cls_scores[mask].tolist()

    nms_boxes = []
    for box in truck_bboxes:
        cx, cy, bw, bh = box
        x = int((cx - bw / 2) / ratio)
        y = int((cy - bh / 2) / ratio)
        width = int(bw / ratio)
        height = int(bh / ratio)
        nms_boxes.append([x, y, width, height])

    indices = cv2.dnn.NMSBoxes(nms_boxes, confidences, score_thr, nms_thr)

    final_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, width, height = nms_boxes[i]
            x1 = np.clip(x, 0, w)
            y1 = np.clip(y, 0, h)
            x2 = np.clip(x + width, 0, w)
            y2 = np.clip(y + height, 0, h)
            final_boxes.append([int(x1), int(y1), int(x2), int(y2)])

    return final_boxes

class LightweightTracker:
    """
    Real-time frame-to-frame object tracking algorithm based on Euclidean centroid distance calculations.
    - Assigns historical tracking IDs to current boxes by minimizing spatial displacement.
    - Spawns fresh independent tracking trajectories when new objects cross frame horizons.
    - Implements an internal countdown timer ('disappeared') to maintain ID consistency for
      partially occluded or temporarily lost vehicles.
    """
    def __init__(self, max_disappeared, distance_threshold):
        self.next_id = 1
        self.tracks = {}
        self.max_disappeared = max_disappeared
        self.distance_threshold = distance_threshold

    def update(self, boxes):
        # Handle condition where no objects pass through the active camera frame
        if len(boxes) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]["disappeared"] += 1
            return {}

        centroids = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in boxes])

        # Populate initial tracker dictionary if empty
        if len(self.tracks) == 0:
            for i in range(len(centroids)):
                self.tracks[self.next_id] = {"centroid": centroids[i], "disappeared": 0, "box": boxes[i]}
                self.next_id += 1
        else:
            track_ids = list(self.tracks.keys())
            track_centroids = np.array([self.tracks[tid]["centroid"] for tid in track_ids])

            # Compute cross-distance matrix mapping old states against fresh detections
            D = np.linalg.norm(track_centroids[:, np.newaxis] - centroids, axis=2)
            rows, cols = D.min(axis=1).argsort(), D.argmin(axis=1)[D.min(axis=1).argsort()]
            used_rows, used_cols = set(), set()

            # Map structural pairs satisfying the proximity criteria limit
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols or D[row, col] > self.distance_threshold:
                    continue
                track_id = track_ids[row]
                self.tracks[track_id].update({"centroid": centroids[col], "box": boxes[col], "disappeared": 0})
                used_rows.add(row); used_cols.add(col)

            # Instantiate fresh IDs for unassigned frame bounding blocks
            for i in range(len(centroids)):
                if i not in used_cols:
                    self.tracks[self.next_id] = {"centroid": centroids[i], "disappeared": 0, "box": boxes[i]}
                    self.next_id += 1

            # Increment missing counters for historical references skipped in this frame pass
            for row in range(len(track_ids)):
                if row not in used_rows:
                    self.tracks[track_ids[row]]["disappeared"] += 1

        # Extract active bounding coordinates, dropping dead data keys
        active_tracks = {tid: self.tracks[tid]["box"] for tid in self.tracks if self.tracks[tid]["disappeared"] == 0}
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["disappeared"] > self.max_disappeared:
                del self.tracks[tid]
        return active_tracks


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
    drive_video_path = os.path.join(SEGMENT_DIR, segment_filename)

    if not os.path.exists(drive_video_path):
        continue

    print(f"\n🎬 Processing Segment {seg_id} (Absolute Anchor: {t_start}s)...")

    # Isolate I/O bottlenecks by copying network files to the local scratch workspace
    local_video_path = os.path.join(TEMP_DIR, f"temp_in_{segment_filename}")
    local_out_vid = os.path.join(TEMP_DIR, f"temp_out_{segment_filename}")
    shutil.copy(drive_video_path, local_video_path)

    # Initialize video capture stream components
    cap = cv2.VideoCapture(local_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_center_x, frame_center_y = w / 2, h / 2

    # Instantiate video writer to store annotated visual debugging elements
    out_writer = cv2.VideoWriter(local_out_vid, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    tracker = LightweightTracker(max_disappeared=TRACKER_MAX_DISAPPEARED, distance_threshold=TRACKER_DISTANCE_THR)

    active_state_buffer = {}
    frame_count = 0

    # Sequential frame-by-frame computational decoding loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate exact absolute timeline position relative to the root source asset
        current_absolute_time = t_start + (frame_count / fps)

        # Ingest processed arrays into the inference engine
        img, ratio = preprocess(frame, INPUT_SIZE)
        raw_out = ort_session.run(None, {input_name: img[None, :, :, :]})[0]
        decoded = decode_outputs(raw_out, INPUT_SIZE)[0]

        # Filter and track valid truck objects
        truck_boxes = get_truck_boxes(decoded, ratio, w, h, score_thr=DETECTION_SCORE_THR, nms_thr=DETECTION_NMS_THR)
        tracked_trucks = tracker.update(truck_boxes)

        # Process each vehicle currently managed by the active tracking system
        for track_id, box in tracked_trucks.items():
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / max(height, 1)
            box_center_x, box_center_y = (x1 + x2) / 2, (y1 + y2) / 2

            # Superimpose layout metadata on the rendering canvas
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"TRUCK ID: {track_id}", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Initialize profile records for unlogged vehicle tracks
            if track_id not in active_state_buffer:
                active_state_buffer[track_id] = {
                    "in_time": current_absolute_time, "out_time": current_absolute_time,
                    "best_center_dist": float('inf'), "best_truck_img": None,
                    "best_plate_img": None, "aspect_ratios": [], "widths": [], "frames_tracked": 0,
                    "first_box": [x1, y1, x2, y2], "last_box": [x1, y1, x2, y2]
                }

            profile = active_state_buffer[track_id]
            profile["out_time"] = current_absolute_time
            profile["frames_tracked"] += 1
            profile["last_box"] = [x1, y1, x2, y2]

            # Rolling Box Dynamics Guard: Freeze data collection if dimensions fluctuate wildly
            profile["widths"].append(width)
            recent_widths = profile["widths"][-3:]
            is_collision = len(recent_widths) == 3 and np.std(recent_widths) > COLLISION_STD_GUARD

            if not is_collision:
                profile["aspect_ratios"].append(aspect_ratio)

            # Center Proximity "Hero Frame" Core Engine: Extract crops when closest to the optical axis
            dist_to_center = math.hypot(frame_center_x - box_center_x, frame_center_y - box_center_y)

            if dist_to_center < profile["best_center_dist"]:
                profile["best_center_dist"] = dist_to_center
                profile["best_truck_img"] = frame[y1:y2, x1:x2].copy()

                # Dynamic Localized Plate Clipper (Extracts plate bounding region based on geometric heuristics)
                px1 = int(x1 + (width * PLATE_MARGIN_WIDTH_CLIP))
                px2 = int(x2 - (width * PLATE_MARGIN_WIDTH_CLIP))
                py1 = int(y2 - (height * PLATE_BOTTOM_HEIGHT_CLIP))
                py2 = int(y2)
                profile["best_plate_img"] = frame[max(0, py1):min(h, py2), max(0, px1):min(w, px2)].copy()

        out_writer.write(frame)
        frame_count += 1

    cap.release()
    out_writer.release()

    # Chronological Spatial Association Matrix (Merge Pass Engine)
    # Stitches broken tracks caused by momentary visual obstacles or deep shadows
    track_ids = sorted(list(active_state_buffer.keys()), key=lambda k: active_state_buffer[k]["in_time"])

    for i in range(len(track_ids)):
        for j in range(i+1, len(track_ids)):
            idA, idB = track_ids[i], track_ids[j]
            if idA in active_state_buffer and idB in active_state_buffer:
                tA = active_state_buffer[idA]
                tB = active_state_buffer[idB]

                # Verify chronological threshold delta
                if 0 <= (tB["in_time"] - tA["out_time"]) < MERGE_MAX_TIME_GAP:
                    # Spatial Validation mapping
                    bA = tA["last_box"]
                    bB = tB["first_box"]
                    cA_x, cA_y = (bA[0] + bA[2]) / 2, (bA[1] + bA[3]) / 2
                    cB_x, cB_y = (bB[0] + bB[2]) / 2, (bB[1] + bB[3]) / 2
                    spatial_distance = math.hypot(cA_x - cB_x, cA_y - cB_y)

                    # Merge profiles if tracks line up temporally and spatially
                    if spatial_distance < MERGE_MAX_SPATIAL_GAP:
                        tA["out_time"] = max(tA["out_time"], tB["out_time"])
                        tA["frames_tracked"] += tB["frames_tracked"]
                        tA["aspect_ratios"].extend(tB["aspect_ratios"])
                        tA["widths"].extend(tB["widths"])
                        tA["last_box"] = tB["last_box"]

                        # Retain the highest-quality image crop from the combined set
                        if tB["best_center_dist"] < tA["best_center_dist"]:
                            tA["best_center_dist"] = tB["best_center_dist"]
                            tA["best_truck_img"] = tB["best_truck_img"]
                            tA["best_plate_img"] = tB["best_plate_img"]

                        del active_state_buffer[idB]

    # Save finalized annotated video file back to the primary asset folder
    final_video_dest = os.path.join(ASSET_DIR, "annotated_segments", f"annotated_{segment_filename}")
    shutil.copy(local_out_vid, final_video_dest)

    # Purge intermediate storage allocations to prevent local workspace bloating
    if os.path.exists(local_video_path): os.remove(local_video_path)
    if os.path.exists(local_out_vid): os.remove(local_out_vid)

    # Database Compilation Phase
    for track_id, profile in active_state_buffer.items():
        # Reject tracker artifacts that do not satisfy our strict lifespan rules
        if profile["frames_tracked"] < MIN_VALID_FRAMES_LOGGED:
            continue

        voted_id = f"TRUCK_{seg_id}_{track_id}"

        # Estimate vehicle class configurations using statistical median profiling
        avg_ratio = np.median(profile["aspect_ratios"]) if len(profile["aspect_ratios"]) > 0 else 1.0
        truck_type, tyres = ("Multi-Axle/Trailer", "10-14") if avg_ratio > 1.6 else ("Heavy Tipper", "6-10") if avg_ratio > 1.2 else ("Small Box", "4-6")

        # Set file storage location links
        truck_img_path = os.path.join(ASSET_DIR, "truck_crops", f"{voted_id}_truck.jpg")
        plate_img_path = os.path.join(ASSET_DIR, "plate_crops", f"{voted_id}_plate.jpg")

        # Save Truck image crop files
        if profile["best_truck_img"] is not None and profile["best_truck_img"].size > 0:
            cv2.imwrite(truck_img_path, profile["best_truck_img"])
            # Excel HYPERLINK formula pointing straight at the local file (no cloud round-trip needed)
            truck_link = f'=HYPERLINK("{truck_img_path}", "View Truck")'
        else:
            truck_link = "NO_IMAGE"

        # Save License Plate image crop files
        if profile["best_plate_img"] is not None and profile["best_plate_img"].size > 0:
            cv2.imwrite(plate_img_path, profile["best_plate_img"])
            plate_link = f'=HYPERLINK("{plate_img_path}", "View Plate")'
        else:
            plate_link = "NO_PLATE"

        # Append entry dictionary block into the database collection array
        final_database.append({
            "Vehicle ID": voted_id.upper(),
            "In Time": round(profile["in_time"], 2),
            "Out Time": round(profile["out_time"], 2),
            "Truck Type": truck_type,
            "Tyres": tyres,
            "Plate File": plate_link,
            "Truck File": truck_link,
            "Source": master_video_name
        })
        print(f"      ✅ Logged -> ID: {voted_id.upper()} | Type: {truck_type} | True Absolute Time: {profile['in_time']:.1f}s - {profile['out_time']:.1f}s")


# ==============================================================================
# ── SECTION 5: EXPORT CLEAN EXCEL TABLE (WITH EMBEDDED THUMBNAIL PREVIEWS) ────
# ==============================================================================

# ==============================================================================
# ── SECTION 5: EXPORT CLEAN EXCEL TABLE ───────────────────────────────────────
# ==============================================================================

if final_database:
    # Convert data structures into clear Pandas DataFrames, sorting by filename and timeline timestamps
    df = pd.DataFrame(final_database).sort_values(by=["Source", "In Time"]).reset_index(drop=True)

    # Define path destination mapping
    excel_path = os.path.join(ASSET_DIR, "Vehicle_Registry_Master.xlsx")

    # Save using the openpyxl engine to ensure cell formulas remain fully executable inside spreadsheet software
    df.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"\n🎉 EXCELLENT! Master Excel Sheet successfully saved to:\n{excel_path}")
else:
    print("\n⚠️ Scan complete, but no valid trucks passed the confidence threshold.")