"""
quantize_int8.py

Runs NNCF post-training quantization on the FP32 OpenVINO IR model, using the
calibration frames from build_calibration_data.py. Produces a SEPARATE
yolox_small_int8.xml/.bin pair — the original FP32 IR is never touched, so you
can A/B compare both at any time by switching MODEL_PRECISION in
step_3_yoloXs_images.py.

Run build_calibration_data.py FIRST — this expects calibration_frames.npy to
already exist. Run create_model_variants.py FIRST too, so weights/yolox_small_fp32.xml
already exists (see that script's docstring for why quantizing from a TRUE
FP32 source matters, vs. the old mislabeled-FP16 file).
"""
import os
import numpy as np
import openvino as ov
import nncf

# CHANGED: yolox_small.xml was being saved via ov.save_model() with no
# compress_to_fp16 override, which defaults to True — so that file was
# actually FP16-compressed, not true FP32. Quantizing from it meant your
# INT8 model was built on top of an already-lossy FP16 baseline, not FP32.
# yolox_small_fp32.xml (from create_model_variants.py, compress_to_fp16=False)
# is the correct true-FP32 source to quantize from.
FP32_IR_PATH = os.path.join("weights", "yolox_small_fp32.xml")
INT8_IR_PATH = os.path.join("weights", "yolox_small_int8.xml")
CALIBRATION_DATA_PATH = "calibration_frames.npy"


def main():
    """
    Loads the true FP32 IR model and the calibration frames array, wraps the
    frames in an nncf.Dataset (with a transform_fn that just adds the batch
    dimension — the frames are already preprocessed exactly like real
    inference input, via build_calibration_data.py's use of the same
    preprocess() function), runs NNCF's post-training INT8 quantization, and
    saves the result as its own separate .xml/.bin pair — never overwriting
    the FP32 source.
    """
    if not os.path.exists(FP32_IR_PATH):
        raise FileNotFoundError(
            f"{FP32_IR_PATH} not found — run create_model_variants.py first "
            f"to generate the true FP32 and FP16 IR files."
        )
    
    if not os.path.exists(CALIBRATION_DATA_PATH):
        raise FileNotFoundError(
            f"{CALIBRATION_DATA_PATH} not found — run build_calibration_data.py first."
        )

    print("Loading FP32 IR model...")
    core = ov.Core()
    model = core.read_model(FP32_IR_PATH)

    print("Loading calibration frames...")
    calibration_frames = np.load(CALIBRATION_DATA_PATH)  # shape (N, 3, 640, 640)

    # NNCF needs a Dataset wrapping an iterable + a transform function that
    # turns each item into exactly what the model's input layer expects —
    # here that's just adding the batch dimension.
    def transform_fn(data_item):
        return data_item[None, :, :, :]  # (3, 640, 640) -> (1, 3, 640, 640)

    calibration_dataset = nncf.Dataset(calibration_frames, transform_fn)

    print(f"Running NNCF post-training quantization on {len(calibration_frames)} calibration frames...")
    quantized_model = nncf.quantize(model, calibration_dataset)

    ov.save_model(quantized_model, INT8_IR_PATH)
    print(f"\n✅ Saved INT8 quantized model to {INT8_IR_PATH}")
    print("Switch MODEL_PRECISION = 'INT8' in step_3_yoloXs_images.py to use it.")


if __name__ == "__main__":
    main()