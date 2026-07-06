# -*- coding: utf-8 -*-
"""
step_3_functions_yoloXs_images.py

Reusable functions/classes pulled out of step_3_yoloXs_images.py.
  - ensure_yolox_weights      <- auto-download guard for the ONNX weights
  - load_yolox_session        <- Section 2 (load the ONNX Runtime session)
  - preprocess                <- Section 3 (unchanged)
  - decode_outputs            <- Section 3 (unchanged)
  - get_truck_boxes           <- Section 3 (unchanged, now takes truck_class_id as a param)
  - LightweightTracker        <- Section 3 (unchanged)
  - run_detection_tracking    <- Section 4's per-segment frame-by-frame detect+track loop
  - merge_broken_tracks       <- Section 4's "Merge Pass Engine"
  - compile_segment_database  <- Section 4's "Database Compilation Phase"
  - save_vehicle_registry     <- Section 5 (Excel export)

REFACTOR NOTE: All print() calls have been replaced with emit() from progress.py
so progress can be streamed to a browser later via FastAPI SSE, while still
printing locally when running from the command line. job_id=None is threaded
through every function that emits — passing None keeps local CLI behavior identical.

REFACTOR NOTE 2 (structured events): every emit() call below now also passes
stage="detection" and a short ui_message for the frontend's narrative UI.
compile_segment_database's per-truck emit() also passes meta={"truck_found": True}
so the frontend can keep a live running count of trucks found without having
to parse the text.
"""
import os
import cv2
import numpy as np
import pandas as pd
import math
import urllib.request
# import onnxruntime as ort
import openvino as ov          # was: import onnxruntime as ort

from progress import emit
# ==============================================================================
# ── ensure_yolox_weights ───────────────────────────────────────────────────────
# ==============================================================================
def ensure_yolox_weights(onnx_path, weights_url, job_id=None):
    """Auto-downloads the YOLOX-S ONNX weights once if they aren't already sitting at onnx_path."""
    if not os.path.exists(onnx_path):
        emit("📥 YOLOX-S weights not found locally — downloading once from GitHub...", job_id=job_id,
             stage="detection", ui_message="Setting up truck detection for the first time...")
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        urllib.request.urlretrieve(weights_url, onnx_path)
        emit(f"✅ Saved weights to {onnx_path}", job_id=job_id,
             stage="detection", ui_message="Detection model ready.")
# ==============================================================================
# ── load_yolox_session ─────────────────────────────────────────────────────────
# ==============================================================================
def load_yolox_session(model_path, job_id=None):
    """
    Initializes an OpenVINO CPU inference session from a model file.

    model_path can be:
      - a .xml IR file (FP32 OR INT8) -> loaded directly, no conversion needed
      - a .onnx file -> converted once to FP32 IR (cached next to it), then loaded

    Returns (infer_request, input_layer, output_layer). infer_request is
    created ONCE here and reused across every frame in run_detection_tracking —
    do not recreate it per frame.
    """
    core = ov.Core()
    ext = os.path.splitext(model_path)[1].lower()

    if ext == ".xml":
        emit(f"⚡ Loading OpenVINO IR directly: {os.path.basename(model_path)}", job_id=job_id,
             stage="detection", ui_message="Loading detection model...")
        model = core.read_model(model_path)

    elif ext == ".onnx":
        ir_path = os.path.splitext(model_path)[0] + ".xml"
        if os.path.exists(ir_path):
            emit("⚡ Loading cached FP32 OpenVINO IR (skips ONNX conversion)...", job_id=job_id,
                 stage="detection", ui_message="Loading detection model...")
            model = core.read_model(ir_path)
        else:
            emit("🔄 First run — converting YOLOX-S ONNX to OpenVINO IR (one-time cost)...", job_id=job_id,
                 stage="detection", ui_message="Preparing detection model for the first time...")
            model = core.read_model(model_path)
            ov.save_model(model, ir_path)  # cache it — future runs load the .xml branch above
    else:
        raise ValueError(f"Unsupported model file type: {model_path}")

    compiled_model = core.compile_model(model, "CPU")
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    infer_request = compiled_model.create_infer_request()

    return infer_request, input_layer, output_layer
# ==============================================================================
# ── preprocess ─────────────────────────────────────────────────────────────────
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
# ==============================================================================
# ── decode_outputs ─────────────────────────────────────────────────────────────
# ==============================================================================
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
# ==============================================================================
# ── get_truck_boxes ────────────────────────────────────────────────────────────
# ==============================================================================
def get_truck_boxes(outputs, ratio, w, h, score_thr, nms_thr, truck_class_id=7):
    """
    Parses structural raw multi-class matrices to extract targets matching the target Truck class.
    - Filters bounding coordinates using a confidence threshold mask.
    - Resolves raw spatial bounding dimensions into standard pixel coordinates.
    - Executes Non-Maximum Suppression (NMS) to eliminate duplicate overlapping bounding boxes.
    - Soft-clips target dimensions to protect against edge out-of-bounds boundary exceptions.
    truck_class_id: COCO dataset index for trucks. DO NOT CHANGE (default 7).
    """
    boxes = outputs[:, :4]
    scores = outputs[:, 4:5] * outputs[:, 5:]
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]
    mask = (cls_scores > score_thr) & (cls_inds == truck_class_id)
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
# ==============================================================================
# ── LightweightTracker ────────────────────────────────────────────────────────
# ==============================================================================
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
# ── run_detection_tracking ─────────────────────────────────────────────────────
# ==============================================================================
def run_detection_tracking(local_video_path, local_out_vid, t_start, infer_request, input_layer, output_layer,
                            input_size, truck_class_id, detection_score_thr, detection_nms_thr,
                            tracker_max_disappeared, tracker_distance_thr, collision_std_guard,
                            plate_margin_width_clip, plate_bottom_height_clip, job_id=None):
    """
    (docstring unchanged from your original — same per-segment detect+track loop)
    """
    cap = cv2.VideoCapture(local_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_center_x, frame_center_y = w / 2, h / 2
    out_writer = cv2.VideoWriter(local_out_vid, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    tracker = LightweightTracker(max_disappeared=tracker_max_disappeared, distance_threshold=tracker_distance_thr)
    active_state_buffer = {}
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_absolute_time = t_start + (frame_count / fps)
        img, ratio = preprocess(frame, input_size)

        # OpenVINO swap: was
        #   raw_out = ort_session.run(None, {input_name: img[None, :, :, :]})[0]
        # infer_request is reused across every frame — created once in load_yolox_session.
        infer_request.infer({input_layer: img[None, :, :, :]})
        # .copy() matters here: the tensor's .data is a view into a buffer that OpenVINO
        # reuses on the NEXT infer() call. decode_outputs() below mutates this array in
        # place, and without a copy, that mutation could corrupt shared buffer state that
        # the next frame's infer() call reads from.
        raw_out = infer_request.get_output_tensor(output_layer.index).data.copy()

        decoded = decode_outputs(raw_out, input_size)[0]

        truck_boxes = get_truck_boxes(decoded, ratio, w, h, score_thr=detection_score_thr,
                                       nms_thr=detection_nms_thr, truck_class_id=truck_class_id)
        tracked_trucks = tracker.update(truck_boxes)

        for track_id, box in tracked_trucks.items():
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / max(height, 1)
            box_center_x, box_center_y = (x1 + x2) / 2, (y1 + y2) / 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"TRUCK ID: {track_id}", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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
            profile["widths"].append(width)
            recent_widths = profile["widths"][-3:]
            is_collision = len(recent_widths) == 3 and np.std(recent_widths) > collision_std_guard
            if not is_collision:
                profile["aspect_ratios"].append(aspect_ratio)
            dist_to_center = math.hypot(frame_center_x - box_center_x, frame_center_y - box_center_y)
            if dist_to_center < profile["best_center_dist"]:
                profile["best_center_dist"] = dist_to_center
                profile["best_truck_img"] = frame[y1:y2, x1:x2].copy()
                px1 = int(x1 + (width * plate_margin_width_clip))
                px2 = int(x2 - (width * plate_margin_width_clip))
                py1 = int(y2 - (height * plate_bottom_height_clip))
                py2 = int(y2)
                profile["best_plate_img"] = frame[max(0, py1):min(h, py2), max(0, px1):min(w, px2)].copy()

        out_writer.write(frame)
        frame_count += 1

    cap.release()
    out_writer.release()
    return active_state_buffer
# ==============================================================================
# ── merge_broken_tracks ────────────────────────────────────────────────────────
# ==============================================================================
def merge_broken_tracks(active_state_buffer, merge_max_time_gap, merge_max_spatial_gap):
    """
    Chronological Spatial Association Matrix (Merge Pass Engine).
    Stitches broken tracks caused by momentary visual obstacles or deep shadows by
    merging any two tracks that line up both temporally and spatially. Mutates and
    returns active_state_buffer with merged-away IDs removed.

    No emit() calls in here — this is pure logic, no status messages needed.
    """
    track_ids = sorted(list(active_state_buffer.keys()), key=lambda k: active_state_buffer[k]["in_time"])
    for i in range(len(track_ids)):
        for j in range(i+1, len(track_ids)):
            idA, idB = track_ids[i], track_ids[j]
            if idA in active_state_buffer and idB in active_state_buffer:
                tA = active_state_buffer[idA]
                tB = active_state_buffer[idB]
                # Verify chronological threshold delta
                if 0 <= (tB["in_time"] - tA["out_time"]) < merge_max_time_gap:
                    # Spatial Validation mapping
                    bA = tA["last_box"]
                    bB = tB["first_box"]
                    cA_x, cA_y = (bA[0] + bA[2]) / 2, (bA[1] + bA[3]) / 2
                    cB_x, cB_y = (bB[0] + bB[2]) / 2, (bB[1] + bB[3]) / 2
                    spatial_distance = math.hypot(cA_x - cB_x, cA_y - cB_y)
                    # Merge profiles if tracks line up temporally and spatially
                    if spatial_distance < merge_max_spatial_gap:
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
    return active_state_buffer
# ==============================================================================
# ── compile_segment_database ───────────────────────────────────────────────────
# ==============================================================================
def compile_segment_database(active_state_buffer, seg_id, master_video_name, asset_dir, min_valid_frames_logged, job_id=None):
    """
    Database Compilation Phase: filters out tracks that don't meet the lifespan rule,
    classifies each truck by aspect ratio, saves its best truck/plate crops to disk,
    builds local-file HYPERLINK formulas, and returns the list of row dicts to log.
    """
    rows = []
    for track_id, profile in active_state_buffer.items():
        # Reject tracker artifacts that do not satisfy our strict lifespan rules
        if profile["frames_tracked"] < min_valid_frames_logged:
            continue
        voted_id = f"TRUCK_{seg_id}_{track_id}"
        # Estimate vehicle class configurations using statistical median profiling
        avg_ratio = np.median(profile["aspect_ratios"]) if len(profile["aspect_ratios"]) > 0 else 1.0
        truck_type, tyres = ("Multi-Axle/Trailer", "10-14") if avg_ratio > 1.6 else ("Heavy Tipper", "6-10") if avg_ratio > 1.2 else ("Small Box", "4-6")
        # Set file storage location links
        truck_img_path = os.path.join(asset_dir, "truck_crops", f"{voted_id}_truck.jpg")
        plate_img_path = os.path.join(asset_dir, "plate_crops", f"{voted_id}_plate.jpg")
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
        rows.append({
            "Vehicle ID": voted_id.upper(),
            "In Time": round(profile["in_time"], 2),
            "Out Time": round(profile["out_time"], 2),
            "Truck Type": truck_type,
            "Tyres": tyres,
            "Plate File": plate_link,
            "Truck File": truck_link,
            "Source": master_video_name
        })
        emit(f"      ✅ Logged -> ID: {voted_id.upper()} | Type: {truck_type} | True Absolute Time: {profile['in_time']:.1f}s - {profile['out_time']:.1f}s", job_id=job_id,
             stage="detection", ui_message=f"Found a {truck_type.lower()} — added to your registry.",
             meta={"truck_found": True})
    return rows
# ==============================================================================
# ── save_vehicle_registry ──────────────────────────────────────────────────────
# ==============================================================================
def save_vehicle_registry(final_database, asset_dir, job_id=None):
    """
    Exports the final_database list to Vehicle_Registry_Master.xlsx inside asset_dir,
    sorted by Source video and In Time.
    """
    if not final_database:
        emit("\n⚠️ Scan complete, but no valid trucks passed the confidence threshold.", job_id=job_id,
             stage="detection", ui_message="No trucks were detected in this footage.")
        # NEW: still write an (empty, headers-only) registry instead of skipping the
        # file entirely. Otherwise the pipeline reports DONE but /download 404s,
        # since it was looking for a file that never got created — even though the
        # run itself succeeded, it just found nothing to log.
        empty_columns = ["Vehicle ID", "In Time", "Out Time", "Truck Type", "Tyres", "Plate File", "Truck File", "Source"]
        df = pd.DataFrame(columns=empty_columns)
        excel_path = os.path.join(asset_dir, "Vehicle_Registry_Master.xlsx")
        df.to_excel(excel_path, index=False, engine='openpyxl')
        emit(f"📄 Saved empty registry (no vehicles detected) to:\n{excel_path}", job_id=job_id,
             stage="detection", ui_message="")  # already said "no trucks" above — nothing new for the UI
        return
    # Convert data structures into clear Pandas DataFrames, sorting by filename and timeline timestamps
    df = pd.DataFrame(final_database).sort_values(by=["Source", "In Time"]).reset_index(drop=True)
    # Define path destination mapping
    excel_path = os.path.join(asset_dir, "Vehicle_Registry_Master.xlsx")
    # Save using the openpyxl engine to ensure cell formulas remain fully executable inside spreadsheet software
    df.to_excel(excel_path, index=False, engine='openpyxl')
    emit(f"\n🎉 EXCELLENT! Master Excel Sheet successfully saved to:\n{excel_path}", job_id=job_id,
         stage="detection", ui_message="Your vehicle registry is ready.")