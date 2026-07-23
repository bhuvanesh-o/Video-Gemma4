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
import re          # <-- add this — needed for _PLATE_CHAR_PATTERN
import cv2
import numpy as np
import pandas as pd
import math
import urllib.request
# import onnxruntime as ort
import openvino as ov          # was: import onnxruntime as ort

from paddleocr import TextDetection, TextRecognition   # was: from paddleocr import PaddleOCR

from progress import emit

# Lazy-loaded singleton — PaddleOCR's detector+recognizer model loads once per
# process, reused across every compile_segment_database() call for the
# lifetime of the job (and across jobs, if they share a process).

_plate_detector = None


# Lazy-loaded singleton — the detector model loads once per process, reused
# across every truck for the lifetime of the job.

_plate_text_detector = None


# Only Latin letters, digits, spaces, hyphens allowed — this is what rejects
# Devanagari/Hindi decorative truck-body text outright, since real plate
# content never contains non-Latin script.
_PLATE_CHAR_PATTERN = re.compile(r'^[A-Za-z0-9\s\-]+$')



# ==============================================================================
# ── compute_frame_quality_score ────────────────────────────────────────────────
# ==============================================================================

def compute_frame_quality_score(crop, x1, y1, x2, y2, frame_w, frame_h,
                                 size_sweet_spot_min, size_sweet_spot_max):
    """
    REFACTOR NOTE (sweet-spot sizing): size used to be a straight bonus —
    bigger box always scored higher. That rewarded the truck's closest,
    biggest, sharpest moment, which (given a fixed side-mounted camera) is
    also the moment the truck is most side-on to the lens — the one point in
    its journey LEAST likely to have a visible front plate. This now scores
    size as a HILL peaking at the midpoint of the sweet-spot window, tapering
    off toward either edge — genuine eligibility filtering (is this frame
    even allowed to compete) happens separately, in run_detection_tracking;
    this score just prefers the center of that window over its edges.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    area = (x2 - x1) * (y2 - y1)
    size_ratio = area / (frame_w * frame_h)

    midpoint = (size_sweet_spot_min + size_sweet_spot_max) / 2
    half_range = (size_sweet_spot_max - size_sweet_spot_min) / 2
    distance_from_center = abs(size_ratio - midpoint)
    size_fitness = max(0.0, 1.0 - (distance_from_center / half_range))  # 1.0 at center, 0.0 at window edges

    touches_edge = (x1 <= 2 or y1 <= 2 or x2 >= frame_w - 2 or y2 >= frame_h - 2)
    edge_penalty = 0.5 if touches_edge else 1.0

    return sharpness * (0.5 + size_fitness) * edge_penalty

'''
def compute_frame_quality_score(crop, x1, y1, x2, y2, frame_w, frame_h):
    """
    Scores a single frame's truck crop on how good a REPRESENTATIVE PHOTO it
    would make — replaces the old "closest to optical center" metric, which
    never actually measured image quality at all.

    Weighted by:
      - Sharpness (Laplacian variance) — motion blur is the single biggest
        killer of plate legibility, so this carries the most weight.
      - Size — bigger box = more pixels of actual detail to work with.
        Secondary factor, since a far-but-sharp truck is still usable.
      - Edge-clipping penalty — a box touching the frame boundary usually
        means the truck is entering/exiting frame and partially cut off.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    area = (x2 - x1) * (y2 - y1)
    size_ratio = area / (frame_w * frame_h)

    touches_edge = (x1 <= 2 or y1 <= 2 or x2 >= frame_w - 2 or y2 >= frame_h - 2)
    edge_penalty = 0.5 if touches_edge else 1.0

    return (sharpness * (1.0 + size_ratio * 2.0)) * edge_penalty
'''
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
# ── insert_spaced_candidate / dedupe_and_trim_candidates ──────────────────────
# ==============================================================================
def insert_spaced_candidate(candidates, new_candidate, max_candidates, min_spacing_frames):
    """
    Inserts a new candidate frame into a truck's shortlist, but ONLY keeps the
    best-scoring frame within any given time neighborhood — same principle as
    the spatial NMS already used on bounding boxes (get_truck_boxes), just
    applied along time instead of space. This is what stops the shortlist
    from filling up with 5 near-duplicate frames from one lucky stretch, while
    still always keeping the best available frame at each rough position
    (not whatever a fixed sampling schedule happened to land on).

    Mutates `candidates` in place, keeps it sorted by score descending.
    """
    for i, existing in enumerate(candidates):
        if abs(existing["frame_index"] - new_candidate["frame_index"]) < min_spacing_frames:
            # Too close in time to an existing candidate — only ONE frame from
            # this neighborhood survives, whichever scores higher.
            if new_candidate["score"] > existing["score"]:
                candidates[i] = new_candidate
                candidates.sort(key=lambda c: c["score"], reverse=True)
            return  # either replaced or discarded — never add as a separate entry

    # No nearby candidate exists — this is a genuinely new time slot.
    if len(candidates) < max_candidates:
        candidates.append(new_candidate)
    else:
        worst_idx = min(range(len(candidates)), key=lambda idx: candidates[idx]["score"])
        if new_candidate["score"] > candidates[worst_idx]["score"]:
            candidates[worst_idx] = new_candidate
        else:
            return
    candidates.sort(key=lambda c: c["score"], reverse=True)


def dedupe_and_trim_candidates(candidates, max_candidates, min_spacing_frames):
    """
    Same spacing rule as insert_spaced_candidate, but applied to a WHOLE list
    at once — used by merge_broken_tracks() when combining two tracks' already-
    built shortlists, where entries need to be re-deduplicated against each
    other rather than inserted one at a time.
    """
    sorted_candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)
    accepted = []
    for c in sorted_candidates:
        if all(abs(c["frame_index"] - a["frame_index"]) >= min_spacing_frames for a in accepted):
            accepted.append(c)
        if len(accepted) >= max_candidates:
            break
    accepted.sort(key=lambda c: c["score"], reverse=True)
    return accepted
    

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
                            max_candidate_frames, size_sweet_spot_min, size_sweet_spot_max,
                            recede_tolerance, min_candidate_spacing_seconds, job_id=None):
    """
    (docstring unchanged from before, plus:)

    REFACTOR NOTE (approach-only + sweet-spot + spaced shortlist): candidate
    frames are now only accepted while (a) the truck's box is still growing
    toward its running peak width — NOT yet past its closest point and
    turning to recede, since a fixed side-mounted camera can't get a usable
    plate view once a truck has passed and is moving away — and (b) the box
    occupies a fraction of the frame within [size_sweet_spot_min,
    size_sweet_spot_max] — too far away is low-detail, too close is
    distorted/near frame edge. Within those bounds, insert_spaced_candidate
    keeps the shortlist spread across genuinely different moments instead of
    letting one lucky stretch dominate all 5 slots.
    """
    cap = cv2.VideoCapture(local_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_writer = cv2.VideoWriter(local_out_vid, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    tracker = LightweightTracker(max_disappeared=tracker_max_disappeared, distance_threshold=tracker_distance_thr)
    active_state_buffer = {}
    frame_count = 0

    # min_candidate_spacing_seconds -> frames, so the same config value behaves
    # consistently across videos with different frame rates.
    min_spacing_frames = max(1, int(fps * min_candidate_spacing_seconds))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_absolute_time = t_start + (frame_count / fps)
        img, ratio = preprocess(frame, input_size)

        infer_request.infer({input_layer: img[None, :, :, :]})
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

            # Extract the crop BEFORE any annotation drawing touches `frame` —
            # cv2.rectangle/putText mutate frame in-place, and drawing first
            # bakes the debug overlay into every saved candidate crop.
            candidate_crop = frame[y1:y2, x1:x2].copy()

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"TRUCK ID: {track_id}", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if track_id not in active_state_buffer:
                active_state_buffer[track_id] = {
                    "in_time": current_absolute_time, "out_time": current_absolute_time,
                    "top_candidates": [],
                    "peak_width": 0,        # running max box width seen so far — proxy for "closest approach"
                    "receding": False,      # flips True permanently once width drops meaningfully below peak
                    "aspect_ratios": [], "widths": [], "frames_tracked": 0,
                    "first_box": [x1, y1, x2, y2], "last_box": [x1, y1, x2, y2]
                }
            profile = active_state_buffer[track_id]
            profile["out_time"] = current_absolute_time
            profile["frames_tracked"] += 1
            profile["last_box"] = [x1, y1, x2, y2]

            profile["widths"].append(width)
            recent_widths = profile["widths"][-3:]
            is_collision = len(recent_widths) == 3 and np.std(recent_widths) > collision_std_guard

            '''
            # TEMP DEBUG
            size_ratio_dbg = (width * height) / (w * h)
            print(f"[DBG] t={current_absolute_time:.1f}s id={track_id} w={width} size_ratio={size_ratio_dbg:.3f} "
                  f"peak={profile['peak_width']} receding={profile['receding']} collision={is_collision}")
            '''

            # Peak-width / receding tracking — this is what detects "has the
            # truck passed its closest point and started moving away."
            if width > profile["peak_width"]:
                profile["peak_width"] = width
            elif profile["peak_width"] > 0 and width < profile["peak_width"] * recede_tolerance:
                profile["receding"] = True  # sticky — once past peak, stays past peak

            if not is_collision:
                profile["aspect_ratios"].append(aspect_ratio)

                size_ratio = (width * height) / (w * h)


                within_sweet_spot = size_sweet_spot_min <= size_ratio <= size_sweet_spot_max
                eligible_for_candidacy = (not profile["receding"]) and within_sweet_spot

                if eligible_for_candidacy and candidate_crop.size > 0:
                    score = compute_frame_quality_score(
                        candidate_crop, x1, y1, x2, y2, w, h,
                        size_sweet_spot_min, size_sweet_spot_max
                    )
                    new_candidate = {"score": score, "crop": candidate_crop, "frame_index": frame_count}
                    insert_spaced_candidate(profile["top_candidates"], new_candidate,
                                             max_candidate_frames, min_spacing_frames)

        out_writer.write(frame)
        frame_count += 1

    cap.release()
    out_writer.release()
    return active_state_buffer

# ==============================================================================
# ── merge_broken_tracks ────────────────────────────────────────────────────────
# ==============================================================================
def merge_broken_tracks(active_state_buffer, merge_max_time_gap, merge_max_spatial_gap,
                         max_candidate_frames, min_candidate_spacing_seconds, fps):
    """
    (docstring unchanged) — now also merges peak_width (keep the larger),
    receding (OR of both — if EITHER half was already past its peak, the
    merged track is considered past peak too), and re-deduplicates the
    combined candidate list with spacing enforcement rather than a raw concat.
    """
    min_spacing_frames = max(1, int(fps * min_candidate_spacing_seconds))
    track_ids = sorted(list(active_state_buffer.keys()), key=lambda k: active_state_buffer[k]["in_time"])
    for i in range(len(track_ids)):
        for j in range(i+1, len(track_ids)):
            idA, idB = track_ids[i], track_ids[j]
            if idA in active_state_buffer and idB in active_state_buffer:
                tA = active_state_buffer[idA]
                tB = active_state_buffer[idB]
                if 0 <= (tB["in_time"] - tA["out_time"]) < merge_max_time_gap:
                    bA = tA["last_box"]
                    bB = tB["first_box"]
                    cA_x, cA_y = (bA[0] + bA[2]) / 2, (bA[1] + bA[3]) / 2
                    cB_x, cB_y = (bB[0] + bB[2]) / 2, (bB[1] + bB[3]) / 2
                    spatial_distance = math.hypot(cA_x - cB_x, cA_y - cB_y)
                    if spatial_distance < merge_max_spatial_gap:
                        tA["out_time"] = max(tA["out_time"], tB["out_time"])
                        tA["frames_tracked"] += tB["frames_tracked"]
                        tA["aspect_ratios"].extend(tB["aspect_ratios"])
                        tA["widths"].extend(tB["widths"])
                        tA["last_box"] = tB["last_box"]

                        tA["peak_width"] = max(tA["peak_width"], tB["peak_width"])
                        tA["receding"] = tA["receding"] or tB["receding"]

                        combined = tA["top_candidates"] + tB["top_candidates"]
                        tA["top_candidates"] = dedupe_and_trim_candidates(
                            combined, max_candidate_frames, min_spacing_frames
                        )

                        del active_state_buffer[idB]
    return active_state_buffer




def get_plate_text_detector(limit_side_len=960, limit_type="max"):
    """
    Standalone PaddleOCR text DETECTION module — deliberately NOT the full
    PaddleOCR() pipeline. See the imports comment above for why: this gets
    us per-box confidence (dt_scores) at roughly half the compute cost of
    running detection + recognition together, since we only need to know
    "is a plate-shaped region here, and how confident," not the decoded text.

    limit_side_len / limit_type: PaddleOCR's own internal resize controls —
    (item #3) this replaces manually downscaling crops ourselves. The
    detection model already resizes its input internally before running;
    limit_type="max" + limit_side_len=960 caps the LARGER side of whatever
    crop comes in at 960px before that internal resize happens. Since your
    truck crops can be huge (a truck filling most of a 4K frame), this stops
    detection cost from scaling with full crop resolution. Lower to 640 for
    more speed if plates are still detected reliably on your footage; raise
    it if small/distant plates start getting missed.

    NOTE: only the FIRST call's limit_side_len/limit_type actually take
    effect, since this is a singleton — the model isn't rebuilt on later
    calls even if you pass different values.

    ITEM #5 (not active yet, documented for later): PaddleOCR ships detection
    models in different size tiers — you're currently on "medium"
    (PP-OCRv6_medium_det). A smaller "mobile"-tier variant, if one exists for
    the v6 line, would trade some accuracy for more speed and is a
    reasonable next lever if 1+2+3 together still aren't fast enough. Verify
    the exact model name for the v6 line (run `paddleocr text_detection
    --help` or check PaddleOCR's docs) before swapping — don't guess the
    string — then change model_name below to try it.
    """
    global _plate_text_detector
    if _plate_text_detector is None:
        _plate_text_detector = TextDetection(
            model_name="PP-OCRv6_medium_det",   # <- item #5 lever: swap to a verified smaller model name here later
            device="cpu",
            enable_mkldnn=False,    # <-- ADDED: same PIR-executor/oneDNN regression you'd
                                    #     already debugged for the old PaddleOCR() pipeline —
                                    #     the standalone TextDetection module apparently
                                    #     defaults to oneDNN on and needed this explicitly too
            limit_side_len=limit_side_len,
            limit_type=limit_type,
        )
    return _plate_text_detector


_plate_text_recognizer = None

def get_plate_text_recognizer():
    """
    Standalone recognition module — used ONLY as a content-based tiebreaker on
    a small shortlist of the strongest detected regions, not on every
    candidate. This is what lets us reject plate-SHAPED, plate-COLORED text
    that still isn't actually a plate (e.g. a painted slogan in a matching
    yellow/white color scheme) — shape and color alone can't catch that,
    only reading the actual characters can.
    """
    global _plate_text_recognizer
    if _plate_text_recognizer is None:
        _plate_text_recognizer = TextRecognition(
            model_name="PP-OCRv6_medium_rec",   # already cached locally from earlier runs
            device="cpu", enable_mkldnn=False,
        )
    return _plate_text_recognizer

def compute_plate_color_score(region):
    """
    Scores how visually consistent a region is with an Indian plate's flat
    yellow (commercial) or white (private) background, versus a truck's
    typically multi-colored painted livery/decorative bumper art (pink, teal,
    yellow-multicolor patterns are common) or a shadow/dark surface.

    Camera-position-independent — works purely off color distribution, not
    where the region sits in frame.
    """
    if region.size == 0:
        return 0.0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    yellow_mask = (h >= 15) & (h <= 35) & (s >= 80) & (v >= 100)
    white_mask = (s <= 40) & (v >= 150)

    return float((yellow_mask | white_mask).mean())


def is_plate_like_text(text, min_alnum_chars=4):
    """Rejects recognized text with any non-Latin/digit/space/hyphen characters
    (catches Devanagari decorative text) or too few real characters (catches
    near-empty/junk recognitions)."""
    if not text:
        return False
    if not _PLATE_CHAR_PATTERN.match(text):
        return False
    return sum(c.isalnum() for c in text) >= min_alnum_chars


# ==============================================================================
# ── locate_plate_via_text_detection ────────────────────────────────────────────
# ==============================================================================
def locate_plate_regions(crop, detector, bottom_fraction=0.6, min_aspect_ratio=1.8, max_aspect_ratio=6.0):
    """
    Returns a list of {"crop": ndarray, "det_score": float} for every
    plate-SHAPED region found — not just the single best one. We need the
    full set here because the highest detection-confidence region isn't
    always the actual plate (a painted slogan can score just as high), so
    downstream scoring (color + recognition) needs multiple candidates to
    choose between, not one pre-committed guess.
    """
    h, w = crop.shape[:2]
    y_start = int(h * (1 - bottom_fraction))
    search_region = crop[y_start:h, :]
    if search_region.size == 0:
        return []

    try:
        output = detector.predict(search_region, batch_size=1)
    except Exception:
        return []

    result = next(iter(output), None)
    if result is None:
        return []

    polys = result.get("dt_polys")
    scores = result.get("dt_scores")
    if polys is None or scores is None or len(polys) == 0:
        return []

    regions = []
    for poly, score in zip(polys, scores):
        xs = [pt[0] for pt in poly]
        ys = [pt[1] for pt in poly]
        bx1, bx2 = int(min(xs)), int(max(xs))
        by1, by2 = int(min(ys)), int(max(ys))
        bw, bh = bx2 - bx1, by2 - by1
        if bh <= 0:
            continue
        if not (min_aspect_ratio <= (bw / bh) <= max_aspect_ratio):
            continue
        pad_x, pad_y = int(bw * 0.1), int(bh * 0.25)
        cx1 = max(0, bx1 - pad_x)
        cx2 = min(search_region.shape[1], bx2 + pad_x)
        cy1 = max(0, by1 - pad_y)
        cy2 = min(search_region.shape[0], by2 + pad_y)
        region_crop = search_region[cy1:cy2, cx1:cx2]
        if region_crop.size > 0:
            regions.append({"crop": region_crop, "det_score": float(score)})

    return regions


# ==============================================================================
# ── select_best_crops ──────────────────────────────────────────────────────────
# ==============================================================================
def select_best_crops(top_candidates, plate_margin_width_clip, plate_bottom_height_clip,
                       plate_confidence_thr=0.5, early_exit_confidence=0.85,
                       color_score_weight=0.4, recognition_check_top_n=2, job_id=None):
    """
    Truck photo: highest quality-scored candidate frame — unchanged.

    Plate crop, in order:
      1. Detect all plate-shaped regions across candidate frames (early-exits
         once a combined shape+color score clears early_exit_confidence).
      2. Score each region by det_score blended with compute_plate_color_score
         — this is what rejects the shadow/wrong-fragment false positives.
      3. Run RECOGNITION only on the top recognition_check_top_n regions —
         this is what rejects plate-shaped, plate-colored text that still
         isn't a real plate (e.g. a Hindi slogan painted in a matching
         yellow/white palette). First region whose recognized text passes
         is_plate_like_text() wins outright.
      4. If nothing passes content validation, fall back to the highest
         combined shape+color score if it clears plate_confidence_thr.
      5. If nothing clears that either, fall back to the old percentage-crop.
    """
    if not top_candidates:
        return None, None

    top_candidates = sorted(top_candidates, key=lambda c: c["score"], reverse=True)
    truck_crop = top_candidates[0]["crop"]

    det_detector = get_plate_text_detector()
    all_regions = []
    for candidate in top_candidates:
        for r in locate_plate_regions(candidate["crop"], det_detector):
            color_score = compute_plate_color_score(r["crop"])
            combined = r["det_score"] * (1 - color_score_weight) + color_score * color_score_weight
            all_regions.append({"crop": r["crop"], "combined_score": combined})
        if all_regions and max(r["combined_score"] for r in all_regions) >= early_exit_confidence:
            break

    if all_regions:
        all_regions.sort(key=lambda r: r["combined_score"], reverse=True)

        recognizer = get_plate_text_recognizer()
        for region in all_regions[:recognition_check_top_n]:
            try:
                rec_output = recognizer.predict(region["crop"], batch_size=1)
                rec_result = next(iter(rec_output), None)
            except Exception:
                continue
            if rec_result is None:
                continue
            rec_text = rec_result.get("rec_text", "")
            rec_score = rec_result.get("rec_score", 0.0)
            if is_plate_like_text(rec_text) and rec_score >= 0.3:
                return truck_crop, region["crop"]   # content-validated — trust immediately

        best = all_regions[0]
        if best["combined_score"] >= plate_confidence_thr:
            if job_id is not None:
                emit("      ⚠️ Plate content validation inconclusive — using best shape/color match.",
                     job_id=job_id, stage="detection", ui_message="")
            return truck_crop, best["crop"]

    # True fallback — nothing found or nothing confident enough.
    height, width = truck_crop.shape[:2]
    fx1 = int(width * plate_margin_width_clip)
    fx2 = int(width * (1 - plate_margin_width_clip))
    fy1 = int(height * (1 - plate_bottom_height_clip))
    fy2 = height
    fallback_crop = truck_crop[fy1:fy2, fx1:fx2]
    if job_id is not None:
        emit("      ⚠️ No confident plate detection — used percentage-crop fallback.",
             job_id=job_id, stage="detection", ui_message="")
    return truck_crop, (fallback_crop if fallback_crop.size > 0 else None)


# ==============================================================================
# ── compile_segment_database ───────────────────────────────────────────────────
# ==============================================================================
def compile_segment_database(active_state_buffer, seg_id, master_video_name, asset_dir, min_valid_frames_logged,
                              plate_margin_width_clip, plate_bottom_height_clip,
                              plate_confidence_thr=0.5, early_exit_confidence=0.85,
                              color_score_weight=0.4, recognition_check_top_n=2, job_id=None):
    """
    (docstring unchanged)
    """
    rows = []
    for track_id, profile in active_state_buffer.items():
        if profile["frames_tracked"] < min_valid_frames_logged:
            continue
        voted_id = f"TRUCK_{seg_id}_{track_id}"
        avg_ratio = np.median(profile["aspect_ratios"]) if len(profile["aspect_ratios"]) > 0 else 1.0
        truck_type, tyres = ("Multi-Axle/Trailer", "10-14") if avg_ratio > 1.6 else ("Heavy Tipper", "6-10") if avg_ratio > 1.2 else ("Small Box", "4-6")

        truck_img_path = os.path.join(asset_dir, "truck_crops", f"{voted_id}_truck.jpg")
        plate_img_path = os.path.join(asset_dir, "plate_crops", f"{voted_id}_plate.jpg")

        truck_crop, plate_crop = select_best_crops(
            profile["top_candidates"], plate_margin_width_clip, plate_bottom_height_clip,
            plate_confidence_thr=plate_confidence_thr, early_exit_confidence=early_exit_confidence,
            color_score_weight=color_score_weight, recognition_check_top_n=recognition_check_top_n,
            job_id=job_id
        )

        if truck_crop is not None and truck_crop.size > 0:
            cv2.imwrite(truck_img_path, truck_crop)
            truck_link = f'=HYPERLINK("{truck_img_path}", "View Truck")'
        else:
            truck_link = "NO_IMAGE"

        if plate_crop is not None and plate_crop.size > 0:
            cv2.imwrite(plate_img_path, plate_crop)
            plate_link = f'=HYPERLINK("{plate_img_path}", "View Plate")'
        else:
            plate_link = "NO_PLATE"

        rows.append({
            "Vehicle ID": voted_id.upper(),
            "In Time": round(profile["in_time"], 2),
            "Out Time": round(profile["out_time"], 2),
            "Truck Type": truck_type,
            "Tyres": tyres,
            "Plate File": plate_link,
            "Truck File": truck_link,
            #"Source": master_video_name
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
        
        #empty_columns = ["Vehicle ID", "In Time", "Out Time", "Truck Type", "Tyres", "Plate File", "Truck File", "Source"]
        empty_columns = ["Vehicle ID", "In Time", "Out Time", "Truck Type", "Tyres", "Plate File", "Truck File"]

        df = pd.DataFrame(columns=empty_columns)
        excel_path = os.path.join(asset_dir, "Vehicle_Registry_Master.xlsx")
        df.to_excel(excel_path, index=False, engine='openpyxl')
        emit(f"📄 Saved empty registry (no vehicles detected) to:\n{excel_path}", job_id=job_id,
             stage="detection", ui_message="")  # already said "no trucks" above — nothing new for the UI
        return
    # Convert data structures into clear Pandas DataFrames, sorting by filename and timeline timestamps
    
    # df = pd.DataFrame(final_database).sort_values(by=["Source", "In Time"]).reset_index(drop=True)
    df = pd.DataFrame(final_database).sort_values(by=["In Time"]).reset_index(drop=True)

    # Define path destination mapping
    excel_path = os.path.join(asset_dir, "Vehicle_Registry_Master.xlsx")
    # Save using the openpyxl engine to ensure cell formulas remain fully executable inside spreadsheet software
    df.to_excel(excel_path, index=False, engine='openpyxl')
    emit(f"\n🎉 EXCELLENT! Master Excel Sheet successfully saved to:\n{excel_path}", job_id=job_id,
         stage="detection", ui_message="Your vehicle registry is ready.")