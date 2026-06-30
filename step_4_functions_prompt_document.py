# -*- coding: utf-8 -*-
"""
step_4_functions_prompt_document.py

Reusable functions pulled out of step_4_prompt_document.py.

  - detect_device            <- CUDA/CPU + dtype auto-detection
  - load_gemma_model         <- Cell 1's model-loading try/except block
  - append_to_report         <- unchanged helper
  - extract_segment_storyboard <- Cell 3's frame-sampling helper (unchanged, with
                                   the hardcoded "cap at 20 frames" now a parameter)
  - run_gemma_inference      <- Cell 3's inner try-block: build inputs, run
                                 model.generate, decode the response
  - purge_gpu_memory         <- Cell 3's finally-block GPU/CPU memory cleanup

NOTE: `scipy.spatial.distance`, `scipy.signal.find_peaks/savgol_filter`, and
`import time` from the original script's imports were never actually used
anywhere in this file — leftover from copy-pasting between notebooks — so
they're dropped here.
"""

import os
import cv2
import numpy as np
import torch
import gc  # Garbage collector for deep memory purging
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM


# ==============================================================================
# ── detect_device ──────────────────────────────────────────────────────────────
# ==============================================================================
def detect_device():
    """Auto-detect hardware: use CUDA + float16 if available, else CPU + float32."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if device == "cpu":
        print("⚠️ No CUDA GPU detected — running Gemma 4 on CPU. This will be slow.")
    return device, dtype


# ==============================================================================
# ── load_gemma_model ───────────────────────────────────────────────────────────
# ==============================================================================
def load_gemma_model(model_id, dtype, device_label="cpu"):
    """
    Fetches open weights securely from cache or repository index.
    Returns (processor, model) on success, or (None, None) if loading fails
    (the error is printed, matching the original notebook's behavior).
    """
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,  # Enforces half-precision float compilation (float16 on GPU, float32 on CPU)
            device_map="auto"
        )
        print(f"✅ Local Gemma 4 engine is loaded and active on execution hardware ({device_label})!")
        return processor, model
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return None, None


# ==============================================================================
# ── append_to_report ───────────────────────────────────────────────────────────
# ==============================================================================
def append_to_report(report_path, text_content):
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(text_content + "\n")


# ==============================================================================
# ── extract_segment_storyboard ─────────────────────────────────────────────────
# ==============================================================================
def extract_segment_storyboard(video_path, interval, max_frames=20):
    """
    Extracts frame snapshots evenly across the ENTIRE pre-cut video segment.
    Converts pixel formats from OpenCV-standard BGR to model-compliant RGB blocks.
    """
    images = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0 or total_frames == 0:
        cap.release()
        return images, 0.0

    # Calculate total duration of this specific segment
    duration = total_frames / fps
    # Establish localized target milestones matching the step interval
    target_seconds = np.arange(0, duration, interval)

    # ⚠️ HARDWARE GUARDRAIL: Adjusts tensor size to prevent VRAM allocation crashes
    if len(target_seconds) > max_frames:
        print(f"  ⚠️ Warning: {len(target_seconds)} frames requested. Capping at {max_frames} to protect VRAM.")
        # Space out max_frames evenly across the segment duration
        target_seconds = np.linspace(0, max(0, duration - 0.1), max_frames)

    for sec in target_seconds:
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)
        ret, frame = cap.read()
        if ret and frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(rgb_frame))

    cap.release()
    return images, duration


# ==============================================================================
# ── run_gemma_inference ────────────────────────────────────────────────────────
# ==============================================================================
def run_gemma_inference(processor, model, snapshot_slideshow, core_prompt, device, dtype):
    """
    Builds the multi-image chat prompt, runs Gemma 4's generate(), and decodes
    the response. Any exception (OOM, bad input, etc.) propagates up so the
    caller's per-segment try/except can log it and move to the next segment.

    Returns the decoded response text (not yet stripped).
    """
    num_frames_extracted = len(snapshot_slideshow)

    # Map structural multi-image structures matching incoming token contexts
    chat_content = [{"type": "image"} for _ in range(num_frames_extracted)]
    chat_content.append({"type": "text", "text": core_prompt})
    messages = [{"role": "user", "content": chat_content}]

    formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=formatted_prompt, images=snapshot_slideshow, return_tensors="pt")

    # Cast floating precision tensors to match the active device's dtype (float16 on GPU, float32 on CPU)
    inputs = {k: v.to(dtype=dtype) if v.is_floating_point() else v for k, v in inputs.items()}
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run tensor evaluation inside non-gradient graph contexts
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1000,
            repetition_penalty=1.2,
            do_sample=True,
            temperature=0.4
        )

    # Isolate generated tokens from user query headers
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = generated_ids[0][input_length:]
    gemma_text = processor.decode(generated_tokens, skip_special_tokens=True)

    return gemma_text


# ==============================================================================
# ── purge_gpu_memory ───────────────────────────────────────────────────────────
# ==============================================================================
def purge_gpu_memory():
    """
    🧹 CRITICAL MEMORY PURGE: Clears GPU/CPU memory so the next segment doesn't
    crash the script.

    NOTE: the original notebook's `del inputs` / `del generated_ids` /
    `del snapshot_slideshow` lines are gone — those tensors now live inside
    run_gemma_inference()'s local scope and get freed automatically the
    moment that function returns, so there's nothing left to explicitly
    delete at this point. gc.collect() + empty_cache() still run the same
    way as before.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
