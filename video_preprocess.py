# -*- coding: utf-8 -*-
"""
video_preprocess.py

Runs ONCE, right after a video is uploaded and BEFORE step_1/step_2/step_3 ever
touch it. Enforces two limits on the raw upload:

  1. DURATION CAP  -> reject anything longer than MAX_DURATION_SECONDS outright.
                      (raises ValueError, so app.py can turn it into a clean
                      error response instead of silently burning compute on a
                      video we're going to refuse anyway)

  2. RESOLUTION CAP -> if the video's height is above MAX_HEIGHT (720p), it gets
                       re-encoded down to 720p (aspect ratio preserved) and the
                       original raw file is OVERWRITTEN with the downscaled copy.
                       If it's already <= 720p, we leave it completely alone —
                       no re-encode, no quality loss, no wasted time.

Because every later stage (step_1 plots, step_2 slices, step_3 YOLOX crops/
annotated segments) always reads from this same raw/ file, downscaling it here
once means the ENTIRE pipeline automatically operates in 720p (or lower) —
nothing in step_1/2/3 needs to know or care.
"""
import os
import cv2

from progress import emit

MAX_DURATION_SECONDS = 240      # hard cap: reject anything longer than 1 minute
MAX_HEIGHT = 720                # resolution ceiling: downscale anything taller than this
MIN_HEIGHT = 720                 # resolution floor: reject anything shorter than this


def get_video_info(video_path):
    """
    Opens a video just long enough to read its metadata (no frame decoding).
    Returns a dict: {duration, width, height, fps, frame_count}.
    Raises ValueError if the file can't be opened or reports invalid metadata.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not open uploaded video (corrupt or unsupported format): {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if not fps or fps <= 0:
        raise ValueError(f"Uploaded video reports invalid FPS (corrupt file?): {video_path}")

    duration = frame_count / fps
    return {
        "duration": duration,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
    }


def _downscale_to_720p(video_path, info, job_id=None):
    """
    Re-encodes video_path down to a height of MAX_HEIGHT (width scaled to keep
    aspect ratio, forced to an even number since some codecs choke on odd
    dimensions), writing to a temp file and then atomically replacing the
    original. Runs entirely locally, same cv2 read/write pattern already used
    elsewhere in this pipeline (step_2's slice_physical_mp4).
    """
    src_width, src_height, fps = info["width"], info["height"], info["fps"]

    scale = MAX_HEIGHT / src_height
    new_width = int(round(src_width * scale))
    new_height = MAX_HEIGHT
    # Force even dimensions — odd widths/heights can break mp4v/H.264 encoders
    if new_width % 2 == 1:
        new_width -= 1
    if new_height % 2 == 1:
        new_height -= 1

    emit(f"📉 Source video is {src_width}x{src_height} — downscaling to {new_width}x{new_height} (720p cap) before processing...", job_id=job_id,
         stage="preparing", ui_message="Optimizing your footage for processing...")

    tmp_path = video_path + ".downscaled_tmp.mp4"
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (new_width, new_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        out.write(resized)

    cap.release()
    out.release()

    # Atomically swap the downscaled file in as the new raw upload
    os.replace(tmp_path, video_path)
    emit(f"✅ Downscale complete — all downstream steps (segments, slices, YOLOX crops) will now use {new_width}x{new_height}.", job_id=job_id,
         stage="preparing", ui_message="Footage ready — starting analysis...")


def validate_and_prepare(video_path, job_id=None):
    """
    Call this ONCE, right after saving the raw upload and BEFORE running
    step_1/step_2/step_3.

    - Raises ValueError (with a user-facing message) if the video is longer
      than MAX_DURATION_SECONDS. Callers should catch this and respond with
      an error instead of starting the background pipeline.
    - If the video's height exceeds MAX_HEIGHT, downscales it in place to
      720p. Otherwise leaves the file untouched.

    Returns the info dict (post-downscale, if a downscale happened) for
    logging/UI purposes.
    """
    info = get_video_info(video_path)

    errors = []
    if info["duration"] > MAX_DURATION_SECONDS:
        errors.append(
            f"Video is {info['duration']:.1f}s long, which exceeds the {MAX_DURATION_SECONDS}s (1 minute) limit."
        )
    if info["height"] < MIN_HEIGHT:
        errors.append(
            f"Video resolution is {info['width']}x{info['height']}, which is below the {MIN_HEIGHT}p minimum."
        )
    if errors:
        raise ValueError(" ".join(errors) + " Please upload a video that is 1 minute or shorter AND 720p or higher.")

    if info["height"] > MAX_HEIGHT:
        _downscale_to_720p(video_path, info, job_id=job_id)
        info = get_video_info(video_path)  # re-read post-downscale dimensions
    else:
        emit(f"✅ Video resolution is {info['width']}x{info['height']} (already 720p) — no downscale needed.", job_id=job_id,
             stage="preparing", ui_message="Footage ready — starting analysis...")

    return info