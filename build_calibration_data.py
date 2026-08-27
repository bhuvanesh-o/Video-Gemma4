"""
build_calibration_data.py

Pulls representative frames from your test videos, runs them through the exact
same preprocess() function step_3 uses at inference time, and saves the result
as a numpy array file (calibration_frames.npy) for quantize_int8.py to consume.

Why frames from test_videos and not random frames: NNCF's calibration step
needs to see the SAME kind of input distribution the model will see in
production — same input_size, same letterbox padding, same pixel value range.
Using anything else (e.g. random noise) would calibrate the quantization
scales against the wrong data and hurt accuracy for no reason.

Run this BEFORE quantize_int8.py.

NOTE (deferred, not yet applied): ideally VIDEO_DIR here should point at a
SEPARATE folder from whatever gets used for post-quantization accuracy
evaluation (e.g. calibration_videos/ vs evaluation_videos/) — otherwise
INT8's benchmark results are partly evaluated on footage it already "saw"
during calibration, which isn't a fully independent test. This is a known,
intentionally-deferred cleanup item, not a bug — VIDEO_DIR is left as
./test_videos for both purposes for now.
"""
import os
import cv2
import numpy as np

from step_3_functions_yoloXs_images import preprocess

INPUT_SIZE = (640, 640)   # must match INPUT_SIZE in step_3_yoloXs_images.py
FRAMES_PER_VIDEO = 20      # frames sampled per video — more videos/frames = better calibration
VIDEO_DIR = "./test_videos"
OUTPUT_PATH = "calibration_frames.npy"


def sample_frames_from_video(video_path, n_frames):
    """
    Opens one video and pulls up to n_frames evenly-spaced frames from across
    its full duration (not just the first few seconds) — via linspace over
    frame indices, then seeking to each one directly with CAP_PROP_POS_FRAMES.
    Returns a list of raw BGR frame arrays (not yet preprocessed).
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    # Evenly spaced indices across the whole video, so calibration sees a
    # representative spread instead of just the first few seconds.
    indices = np.linspace(0, total_frames - 1, min(n_frames, total_frames), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def main():
    """
    Walks every video in VIDEO_DIR, samples FRAMES_PER_VIDEO frames from each,
    runs every frame through the SAME preprocess() letterbox/normalize
    pipeline step_3 uses at real inference time, stacks everything into one
    big numpy array, and saves it to OUTPUT_PATH for quantize_int8.py.
    """
    video_extensions = (".mp4", ".avi", ".mov", ".mkv")
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(video_extensions)]

    if not video_files:
        print(f"No videos found in {VIDEO_DIR}")
        return

    calibration_inputs = []
    for video_name in video_files:
        video_path = os.path.join(VIDEO_DIR, video_name)
        print(f"Sampling frames from {video_name}...")
        frames = sample_frames_from_video(video_path, FRAMES_PER_VIDEO)
        for frame in frames:
            img, _ = preprocess(frame, INPUT_SIZE)  # same letterbox step used at real inference
            calibration_inputs.append(img)

    calibration_array = np.stack(calibration_inputs)  # shape: (N, 3, 640, 640)
    np.save(OUTPUT_PATH, calibration_array)
    print(f"\nSaved {len(calibration_inputs)} calibration frames to {OUTPUT_PATH}")
    print(f"Shape: {calibration_array.shape}")


if __name__ == "__main__":
    main()