# -*- coding: utf-8 -*-
"""
step_4_prompt_document.py

In this we will get the GEMMA 4 model from HUGGING FACE and then we will pass
the VIDEO SEGMENTS into the model by following the PROMPT and then we will
just display the output for each segment based on the prompt.

REFACTOR NOTE: The long processing chunks (device detection, model loading,
frame storyboard extraction, the actual Gemma inference call, GPU memory
cleanup) now live in step_4_functions_prompt_document.py as callable functions. This
file just holds your config/prompt and orchestrates the per-segment loop by
calling into that library.

NOTE: Before running, in your terminal:
    pip install opencv-python numpy pillow torch transformers

(scipy is no longer required — see the lib file's docstring for why.)

⚠️ GPU REQUIREMENT — READ THIS BEFORE RUNNING LOCALLY:
This script loads a multimodal LLM (Gemma 4 E2B). detect_device() in the lib
auto-picks CUDA+float16 if you have an NVIDIA GPU, otherwise CPU+float32. A
Ryzen 5 laptop with no discrete NVIDIA GPU will fall back to CPU, which will
be VERY slow for a multimodal model with up to 20 image frames per segment —
expect minutes per segment, not seconds. For real throughput, keep this
particular step running in Colab and only run the others locally.

Make sure step_4_functions_prompt_document.py is in the same folder as this script
(or somewhere on your PYTHONPATH) so the import below resolves.
"""

import os

from step_4_functions_prompt_document import (
    detect_device,
    load_gemma_model,
    append_to_report,
    extract_segment_storyboard,
    run_gemma_inference,
    purge_gpu_memory,
)

# 🔧 EDIT THESE to point at your local equivalents of the Drive folders
VIDEO_DIR = r"D:\Traffic_Control\trial_video_segments"
REPORT_FILE_PATH = r"D:\Traffic_Control\LLM_Analysis_Report_Gemma4.txt"

if not os.path.exists(VIDEO_DIR):
    raise FileNotFoundError(f"⚠️ Could not find folder: {VIDEO_DIR}")

DEVICE, DTYPE = detect_device()

# Fetch open weights securely from cache or repository index
MODEL_ID = "google/gemma-4-e2b-it"
print(f"🤖 Fetching open weights for {MODEL_ID} to notebook memory...")
processor, model = load_gemma_model(MODEL_ID, DTYPE, device_label=DEVICE)
if model is None:
    raise RuntimeError(f"Could not load {MODEL_ID} — see the error printed above.")

# ==============================================================================
# ── PROMPT & BEHAVIOR CONTROL PANEL (YOUR TESTING GROUND) ────────────────────
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
# ── MASTER FILE TRACKING & INFERENCE LOOP ─────────────────────────────────────
# ==============================================================================
video_extensions = ('.mp4', '.avi', '.mov', '.mkv')

video_paths = []
for subfolder in sorted(os.listdir(VIDEO_DIR)):
    subfolder_path = os.path.join(VIDEO_DIR, subfolder)
    if os.path.isdir(subfolder_path):
        for f in sorted(os.listdir(subfolder_path)):
            if f.lower().endswith(video_extensions):
                video_paths.append(os.path.join(subfolder_path, f))

print(f"📦 Found {len(video_paths)} pre-cut segments in {VIDEO_DIR}.")

for idx, v_path in enumerate(video_paths, 1):
    v_name = os.path.basename(v_path)
    print(f"\n==========================================================")
    print(f"🎬 PROCESSING SEGMENT {idx}/{len(video_paths)}: {v_name}")
    print(f"==========================================================")

    snapshot_slideshow, duration = extract_segment_storyboard(v_path, FRAME_INTERVAL_SECONDS)
    num_frames_extracted = len(snapshot_slideshow)

    if num_frames_extracted == 0:
        print(f"  ❌ Failed to pull visual array maps. Skipping.")
        continue

    print(f"  📸 Pulled {num_frames_extracted} frames over {duration:.1f}s segment...")
    print(f"  🧠 Processing tensor mappings into Gemma 4...")

    try:
        gemma_text = run_gemma_inference(processor, model, snapshot_slideshow, CORE_PROMPT, DEVICE, DTYPE)

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
        purge_gpu_memory()

print("\n🎉 COMPLETE! Sandbox processing run concluded across folder files.")
