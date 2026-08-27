# -*- coding: utf-8 -*-
"""
create_model_variants.py

One-time (or re-run-when-model-changes) script: converts the YOLOX-S ONNX
weights into three clearly-named, correctly-precision-labeled IR variants.

REASON THIS EXISTS: ov.save_model()'s compress_to_fp16 defaults to True.
The old load_yolox_session() called ov.save_model(model, ir_path) with no
override -- meaning the file that's been living at weights/yolox_small.xml
and getting called "FP32" in benchmarks has actually been FP16-compressed
the whole time. This script makes the precision explicit and unambiguous
in both the code and the filename.

NOTE: this script's output (yolox_small_fp32.xml, yolox_small_fp16.xml) is
NOT currently wired into step_3_yoloXs_images.py's MODEL_PRECISION switch —
that file still uses the original two-option ONNX_PATH/INT8_PATH setup by
deliberate choice, not because this script is broken. Run this file and
quantize_int8.py standalone, on their own, when you actually want to
produce/compare true FP32 vs FP16 vs INT8 benchmark numbers.

Run this manually once (or after any model change):
    python create_model_variants.py
"""

import os
import openvino as ov

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
ONNX_PATH = os.path.join(WEIGHTS_DIR, "yolox_small.onnx")

FP32_PATH = os.path.join(WEIGHTS_DIR, "yolox_small_fp32.xml")
FP16_PATH = os.path.join(WEIGHTS_DIR, "yolox_small_fp16.xml")

def main():

    """
    Converts the raw ONNX weights to OpenVINO IR exactly once (ov.convert_model
    is precision-neutral — it doesn't compress anything on its own), then
    saves that same in-memory model twice: once forcing true FP32
    (compress_to_fp16=False), once forcing FP16 (compress_to_fp16=True) —
    so both files are unambiguously and correctly labeled, unlike the old
    single .xml file that silently defaulted to FP16 while being called FP32.
    """
    
    if not os.path.exists(ONNX_PATH):
        raise FileNotFoundError(
            f"{ONNX_PATH} not found -- run the pipeline once first so "
            f"ensure_yolox_weights() downloads it, or place it manually."
        )

    print(f"Converting {ONNX_PATH} -> OpenVINO IR (once, precision-neutral)...")
    model = ov.convert_model(ONNX_PATH)

    print(f"Saving TRUE FP32 -> {FP32_PATH}")
    ov.save_model(model, FP32_PATH, compress_to_fp16=False)

    print(f"Saving FP16 -> {FP16_PATH}")
    ov.save_model(model, FP16_PATH, compress_to_fp16=True)

    print("\nDone. Next: run quantize_int8.py against yolox_small_fp32.xml "
          "to produce yolox_small_int8.xml from the true FP32 baseline.")

if __name__ == "__main__":
    main()