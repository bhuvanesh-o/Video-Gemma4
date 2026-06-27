# -*- coding: utf-8 -*-
"""
Gemma_Prompt_document.py

Ported from Google Colab (Gemma_Prompt_document.ipynb) to a standalone local
script for VS Code.

In this we will get the GEMMA 4 model from HUGGING FACE and then we will pass
the VIDEO SEGMENTS into the model by following the PROMPT and then we will
just display the output for each segment based on the prompt.

NOTE: Before running, in your terminal:
    pip install opencv-python numpy pillow torch transformers scipy

NOTE: `!pip install google-genai` from the original notebook is gone — it was
never actually imported/used anywhere in this script (the model is loaded
locally via `transformers`, not the Gemini API), so it's safe to drop.

⚠️ GPU REQUIREMENT — READ THIS BEFORE RUNNING LOCALLY:
This script loads a multimodal LLM (Gemma 4 E2B) and the original notebook
hardcodes `.to("cuda")` everywhere, assuming a Colab GPU. A Ryzen 5 laptop
with no discrete NVIDIA GPU has no CUDA device, so that line would crash
immediately. Below, device/dtype are now auto-detected:
  - If CUDA is available -> uses it with float16 (fast, as in the original).
  - If not -> falls back to CPU with float32 (this will be VERY slow for a
    multimodal model with up to 20 image frames per segment — expect minutes
    per segment, not seconds). For real throughput, keep this particular
    step running in Colab and only port the others.
"""

# ==============================================================================
# ── CELL 1: MASTER SETUP & LOCAL MULTIMODAL MODEL INITIALIZATION ──────────────
# ==============================================================================
import cv2
import numpy as np
import os
import time
import torch
import scipy.spatial.distance as dist
from scipy.signal import find_peaks, savgol_filter
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM

# 🔧 EDIT THESE to point at your local equivalents of the Drive folders
VIDEO_DIR = r"C:\Users\YourName\GEMMA\trial_main_video\trial_video_segments\truck_video_trial"
REPORT_FILE_PATH = r"C:\Users\YourName\GEMMA\trial_main_video\LLM_Analysis_Report_Gemma4.txt"

if not os.path.exists(VIDEO_DIR):
    raise FileNotFoundError(f"⚠️ Could not find folder: {VIDEO_DIR}")

# Auto-detect hardware: use CUDA + float16 if available, else CPU + float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
if DEVICE == "cpu":
    print("⚠️ No CUDA GPU detected — running Gemma 4 on CPU. This will be slow.")

# Fetch open weights securely from cache or repository index
MODEL_ID = "google/gemma-4-e2b-it"
print(f"🤖 Fetching open weights for {MODEL_ID} to notebook memory...")
try:
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,  # Enforces half-precision float compilation (float16 on GPU, float32 on CPU)
        device_map="auto"
    )
    print(f"✅ Local Gemma 4 engine is loaded and active on execution hardware ({DEVICE})!")
except Exception as e:
    print(f"❌ Error loading weights: {e}")

def append_to_report(report_path, text_content):
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(text_content + "\n")

# ==============================================================================
# ── CELL 2: PROMPT & BEHAVIOR CONTROL PANEL (YOUR TESTING GROUND) ─────────────
# ==============================================================================

# Change this float parameter dynamically to alter how many frames are sampled per second!
# Examples: 1.0 (every second), 0.5 (every half-second), 2.0 (every 2 seconds)
FRAME_INTERVAL_SECONDS = 0.5

# Change, expand, or adjust this text strategy freely to look for different events
CORE_PROMPT = (
    f"Watch this sequence of chronological frames taken at {FRAME_INTERVAL_SECONDS}-second intervals from a CCTV segment. "
    "Analyze the chronological progression of events carefully from first to last frame. "
    "Provide a detailed, structured description answering the following parameters exactly:\n\n"
    "1. OVERALL EVENT: Summarize what action or event is taking place in the scene.\n"
    "2. PEOPLE & ACTIONS: Identify every person visible. What are they doing? "
    "Classify them explicitly as a man, woman, or child. Describe what color clothes/dress they are wearing.\n"
    "3. VEHICLES & ANIMALS: Look closely for any vehicles (specify type: car, van, truck, bike) or animals (specify: dog, cat, etc.). "
    "Note their colors.\n"
    "4. MOVEMENT DIRECTION: For every moving entity (person, vehicle, or animal), state their direction of travel "
    "(e.g., left to right, right to left, moving toward the camera, or moving away from the camera).\n\n"
    "Be precise, objective, and focus purely on active visual movement."
)

print(f"⚙️ Sandbox settings registered! Frame step rate assigned: {FRAME_INTERVAL_SECONDS}s.")

# ==============================================================================
# ── CELL 3: DIRECT SEGMENT INFERENCE & MULTI-IMAGE LLM ENGINE ─────────────────
# ==============================================================================
"""
Core System Description:
This architectural module acts as a localized vision-language inference engine.
It ingests pre-sliced video segments, dynamically extracts storyboard frames
based on the user's requested interval, and pipes the multi-image tensor arrays
directly into a local multi-modal LLM (such as Gemma/PaliGemma) running on a
local CUDA accelerator. It features aggressive VRAM garbage collection to prevent
Out-Of-Memory (OOM) crashes across large datasets.
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
import gc  # Garbage collector for deep memory purging

def extract_segment_storyboard(video_path, interval):
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
    if len(target_seconds) > 20:
        print(f"  ⚠️ Warning: {len(target_seconds)} frames requested. Capping at 20 to protect VRAM.")
        # Space out 20 frames evenly across the segment duration
        target_seconds = np.linspace(0, max(0, duration - 0.1), 20)

    for sec in target_seconds:
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)
        ret, frame = cap.read()
        if ret and frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(rgb_frame))

    cap.release()
    return images, duration

# ── MASTER FILE TRACKING & INFERENCE LOOP ─────────────────────────────────────
all_files = os.listdir(VIDEO_DIR)
video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
# Sorted to process segment_1, segment_2 in chronological order
video_paths = sorted([f for f in all_files if f.lower().endswith(video_extensions)])

print(f"📦 Found {len(video_paths)} pre-cut segments in {VIDEO_DIR}.")

for idx, v_name in enumerate(video_paths, 1):
    print(f"\n==========================================================")
    print(f"🎬 PROCESSING SEGMENT {idx}/{len(video_paths)}: {v_name}")
    print(f"==========================================================")

    v_path = os.path.join(VIDEO_DIR, v_name)

    # 1. Extract frames directly from the physical segment
    snapshot_slideshow, duration = extract_segment_storyboard(v_path, FRAME_INTERVAL_SECONDS)
    num_frames_extracted = len(snapshot_slideshow)

    if num_frames_extracted == 0:
        print(f"  ❌ Failed to pull visual array maps. Skipping.")
        continue

    print(f"  📸 Pulled {num_frames_extracted} frames over {duration:.1f}s segment...")
    print(f"  🧠 Processing tensor mappings into Gemma 4...")

    try:
        # Map structural multi-image structures matching incoming token contexts
        chat_content = [{"type": "image"} for _ in range(num_frames_extracted)]
        chat_content.append({"type": "text", "text": CORE_PROMPT})
        messages = [{"role": "user", "content": chat_content}]

        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=formatted_prompt, images=snapshot_slideshow, return_tensors="pt")

        # Cast floating precision tensors to match the active device's dtype (float16 on GPU, float32 on CPU)
        inputs = {k: v.to(dtype=DTYPE) if v.is_floating_point() else v for k, v in inputs.items()}
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

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

        # Format and save the result
        segment_result = (
            f"📍 Source Segment: {v_name} | Duration: {duration:.1f}s | Sampled {num_frames_extracted} frames\n"
            f"📝 Local Gemma 4 Summary:\n{gemma_text.strip()}\n"
            f"----------------------------------------------------------\n"
        )

        print(f"  ✅ SUCCESS (Segment Logged):")
        print(segment_result)
        append_to_report(REPORT_FILE_PATH, segment_result)

    except Exception as e:
        print(f"  ❌ Local system processing crash on segment {v_name}: {e}")

    finally:
        # 🧹 CRITICAL MEMORY PURGE: Clears GPU/CPU memory so the next segment doesn't crash the script
        if 'inputs' in locals(): del inputs
        if 'generated_ids' in locals(): del generated_ids
        if 'snapshot_slideshow' in locals(): del snapshot_slideshow
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print("\n🎉 COMPLETE! Sandbox processing run concluded across folder files.")