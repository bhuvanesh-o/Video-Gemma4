# benchmark_videos.py

"""
Runs the full GEMMA traffic-analysis pipeline:

    preprocess
        ->
    segmentation
        ->
    slicing
        ->
    detection (YOLOX + PaddleOCR)
        ->
    description (OpenRouter 26B + local E2B LiteRT)

across a folder of benchmark videos.

Records:

    - stage execution time
    - average CPU
    - peak CPU
    - average process-tree RAM
    - peak process-tree RAM
    - Docker container RAM when available

IMPORTANT:

At this level:

    Detection RAM
        = Step 3 as a whole
        = YOLOX + PaddleOCR + Python overhead

    Description RAM
        = Step 4 as a whole
        = API-client overhead + E2B LiteRT + Python overhead

Separating YOLOX, PaddleOCR and E2B individually requires
instrumentation inside Step 3 and Step 4.
"""

import os
import csv
import time
import uuid
import argparse

from dotenv import load_dotenv

from resource_monitor import ResourceMonitor

import storage
import video_preprocess
import step_1_segmentation
import step_2_video_slice_excel_timestamp
import step_3_yoloXs_images
import step_4_prompt_document


load_dotenv()


# ======================================================================
# RESOURCE HELPERS
# ======================================================================

def summarize_monitor(monitor):
    """
    Peak values now come from ResourceMonitor's EXACT kernel-tracked
    high-water marks when available (inside Docker/Linux) -- falling back
    to sampled-max ONLY on native Windows, where no exact source exists.
    """
    if not monitor.samples:
        return {
            "cpu_avg_pct": 0, "cpu_peak_pct": 0,
            "ram_avg_mb": 0, "ram_peak_mb": 0,
            "container_ram_avg_mb": None, "container_ram_peak_mb": None,
        }

    cpu_values = [sample[1] for sample in monitor.samples]
    ram_values = [sample[2] for sample in monitor.samples]
    container_ram_values = [
        sample[3] for sample in monitor.samples
        if len(sample) > 3 and sample[3] is not None
    ]

    result = {
        "cpu_avg_pct": round(sum(cpu_values) / len(cpu_values), 1),
        "cpu_peak_pct": round(max(cpu_values), 1),
        "ram_avg_mb": round(sum(ram_values) / len(ram_values), 1),
        "container_ram_avg_mb": None,
        "container_ram_peak_mb": None,
    }

    # EXACT process-tree peak (Linux VmHWM sum) when available.
    # Falls back to sampled max ONLY when running natively (Windows).
    exact_tree_peak = getattr(monitor, "exact_peak_process_tree_ram_mb", 0)
    result["ram_peak_mb"] = round(exact_tree_peak, 1) if exact_tree_peak > 0 else round(max(ram_values), 1)

    if container_ram_values:
        result["container_ram_avg_mb"] = round(
            sum(container_ram_values) / len(container_ram_values), 1
        )

    exact_container_peak = getattr(monitor, "exact_peak_container_ram_mb", None)
    if exact_container_peak is not None:
        result["container_ram_peak_mb"] = round(exact_container_peak, 1)
    elif container_ram_values:
        result["container_ram_peak_mb"] = round(max(container_ram_values), 1)

    return result




def format_ram(mb):
    if mb is None:
        return "N/A"

    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"

    return f"{mb:.1f} MB"




def print_resource_benchmark(
    stage_times,
    stage_resources,
    overall_resources,
):
    """
    Pretty console report.
    """

    print()
    print("=" * 78)
    print("                     RESOURCE BENCHMARK")
    print("=" * 78)

    print(
        f"{'Stage':<26}"
        f"{'Time':>10}"
        f"{'Avg RAM':>14}"
        f"{'Peak RAM':>14}"
        f"{'CPU Avg':>12}"
    )

    print("-" * 78)

    display_names = {
        "preprocess": "Preprocessing",
        "segmentation": "Segmentation",
        "slicing": "Slicing",
        "detection": "Detection (YOLOX+OCR)",
        "description": "Description (26B+E2B)",
    }

    order = [
        "preprocess",
        "segmentation",
        "slicing",
        "detection",
        "description",
    ]

    for key in order:
        if key not in stage_times:
            continue

        resources = stage_resources[key]

        print(
            f"{display_names[key]:<26}"
            f"{stage_times[key]:>9.2f}s"
            f"{format_ram(resources['ram_avg_mb']):>14}"
            f"{format_ram(resources['ram_peak_mb']):>14}"
            f"{resources['cpu_avg_pct']:>11.1f}%"
        )

    print("-" * 78)

    print(
        f"{'Overall process-tree peak RAM:':<45}"
        f"{format_ram(overall_resources['ram_peak_mb']):>20}"
    )

    print(
        f"{'Docker container peak RAM:':<45}"
        f"{format_ram(overall_resources['container_ram_peak_mb']):>20}"
    )

    print("=" * 78)
    print()

    if overall_resources["container_ram_peak_mb"] is None:
        print(
            "ℹ️ Docker container RAM = N/A because this benchmark "
            "was not running inside a Linux/Docker cgroup."
        )
        print()


# ======================================================================
# PIPELINE HELPERS
# ======================================================================

def count_trucks_found(job_id):
    import pandas as pd

    excel_path = storage.path_for(
        job_id,
        "final_assets",
        "Vehicle_Registry_Master.xlsx",
    )

    if not os.path.exists(excel_path):
        return 0

    return len(
        pd.read_excel(excel_path)
    )


# ======================================================================
# RUN ONE VIDEO
# ======================================================================

def run_one_video(
    video_path,
    keep_outputs=False,
    precision="INT8",
    hint="LATENCY",
):
    video_name = os.path.basename(video_path)

    job_id = (
        f"bench_{uuid.uuid4().hex[:8]}"
    )

    stage_times = {}
    stage_resources = {}

    # ------------------------------------------------------------------
    # OVERALL RESOURCE MONITOR
    # ------------------------------------------------------------------

    overall_monitor = ResourceMonitor(
        interval=0.2
    )

    overall_monitor.start()

    overall_start = time.perf_counter()

    # ------------------------------------------------------------------
    # SAVE INPUT
    # ------------------------------------------------------------------

    with open(video_path, "rb") as f:
        saved_path = storage.save_upload(
            job_id,
            f,
            filename=video_name,
        )

    raw_info = video_preprocess.get_video_info(
        saved_path
    )

    # ==================================================================
    # PREPROCESSING
    # ==================================================================

    preprocess_monitor = ResourceMonitor(
        interval=0.2
    )

    preprocess_monitor.start()

    t0 = time.perf_counter()

    try:
        video_preprocess.validate_and_prepare(
            saved_path,
            job_id=job_id,
        )

    except ValueError as e:
        print(
            f"  ⚠️ Skipped {video_name}: {e}"
        )

        preprocess_monitor.stop()
        overall_monitor.stop()

        if not keep_outputs:
            storage.delete_job(job_id)

        return None

    stage_times["preprocess"] = (
        time.perf_counter() - t0
    )

    preprocess_monitor.stop()

    stage_resources["preprocess"] = (
        summarize_monitor(
            preprocess_monitor
        )
    )

    # ==================================================================
    # STEP 1 — SEGMENTATION
    # ==================================================================

    segmentation_monitor = ResourceMonitor(
        interval=0.2
    )

    segmentation_monitor.start()

    t0 = time.perf_counter()

    step_1_segmentation.main(
        job_id=job_id
    )

    stage_times["segmentation"] = (
        time.perf_counter() - t0
    )

    segmentation_monitor.stop()

    stage_resources["segmentation"] = (
        summarize_monitor(
            segmentation_monitor
        )
    )

    # ==================================================================
    # STEP 2 — SLICING
    # ==================================================================

    slicing_monitor = ResourceMonitor(
        interval=0.2
    )

    slicing_monitor.start()

    t0 = time.perf_counter()

    step_2_video_slice_excel_timestamp.main(
        job_id=job_id
    )

    stage_times["slicing"] = (
        time.perf_counter() - t0
    )

    slicing_monitor.stop()

    stage_resources["slicing"] = (
        summarize_monitor(
            slicing_monitor
        )
    )

    # ==================================================================
    # STEP 3 — DETECTION
    #
    # Currently this measures:
    #
    #     YOLOX
    #     +
    #     PaddleOCR
    #     +
    #     Step-3 Python overhead
    #
    # To split YOLOX and PaddleOCR individually, instrumentation needs
    # to be added INSIDE step_3_yoloXs_images.py.
    # ==================================================================

    detection_monitor = ResourceMonitor(
        interval=0.2
    )

    detection_monitor.start()

    t0 = time.perf_counter()

    step_3_yoloXs_images.main(
        job_id=job_id,
        model_precision=precision,
        performance_hint=hint,
    )

    stage_times["detection"] = (
        time.perf_counter() - t0
    )

    detection_monitor.stop()

    stage_resources["detection"] = (
        summarize_monitor(
            detection_monitor
        )
    )

    trucks_found = count_trucks_found(
        job_id
    )

    # ==================================================================
    # STEP 4 — DESCRIPTION
    #
    # ResourceMonitor tracks child processes recursively.
    #
    # Therefore:
    #
    # Python
    #    ↓
    # script
    #    ↓
    # litert-lm
    #
    # are included in this RAM measurement.
    #
    # But this entire block is still Step 4 as a whole.
    # ==================================================================

    description_monitor = ResourceMonitor(
        interval=0.2
    )

    description_monitor.start()

    t0 = time.perf_counter()

    step_4_prompt_document.main(
        job_id=job_id
    )

    stage_times["description"] = (
        time.perf_counter() - t0
    )

    description_monitor.stop()

    stage_resources["description"] = (
        summarize_monitor(
            description_monitor
        )
    )

    # ==================================================================
    # OVERALL
    # ==================================================================

    total_time = (
        time.perf_counter()
        - overall_start
    )

    overall_monitor.stop()

    overall_resources = (
        summarize_monitor(
            overall_monitor
        )
    )

    duration = raw_info["duration"]

    realtime_factor = (
        total_time / duration
        if duration > 0
        else 0
    )

    video_sec_per_processing_sec = (
        duration / total_time
        if total_time > 0
        else 0
    )

    # ==================================================================
    # PRINT NICE RESOURCE REPORT
    # ==================================================================

    print_resource_benchmark(
        stage_times=stage_times,
        stage_resources=stage_resources,
        overall_resources=overall_resources,
    )

    # ==================================================================
    # RESULT ROW
    # ==================================================================

    result = {
        "video_name": video_name,
        "model_precision": precision,
        "performance_hint": hint,

        "duration_sec": round(
            duration,
            2
        ),

        "resolution": (
            f"{raw_info['width']}x"
            f"{raw_info['height']}"
        ),

        # --------------------------------------------------------------
        # TIME
        # --------------------------------------------------------------

        "time_preprocess_sec": round(
            stage_times["preprocess"],
            2
        ),

        "time_segmentation_sec": round(
            stage_times["segmentation"],
            2
        ),

        "time_slicing_sec": round(
            stage_times["slicing"],
            2
        ),

        "time_detection_sec": round(
            stage_times["detection"],
            2
        ),

        "time_description_sec": round(
            stage_times["description"],
            2
        ),

        "time_total_sec": round(
            total_time,
            2
        ),

        # --------------------------------------------------------------
        # PREPROCESSING
        # --------------------------------------------------------------

        "cpu_preprocess_avg_pct":
            stage_resources["preprocess"]["cpu_avg_pct"],

        "cpu_preprocess_peak_pct":
            stage_resources["preprocess"]["cpu_peak_pct"],

        "ram_preprocess_avg_mb":
            stage_resources["preprocess"]["ram_avg_mb"],

        "ram_preprocess_peak_mb":
            stage_resources["preprocess"]["ram_peak_mb"],

        # --------------------------------------------------------------
        # SEGMENTATION
        # --------------------------------------------------------------

        "cpu_segmentation_avg_pct":
            stage_resources["segmentation"]["cpu_avg_pct"],

        "cpu_segmentation_peak_pct":
            stage_resources["segmentation"]["cpu_peak_pct"],

        "ram_segmentation_avg_mb":
            stage_resources["segmentation"]["ram_avg_mb"],

        "ram_segmentation_peak_mb":
            stage_resources["segmentation"]["ram_peak_mb"],

        # --------------------------------------------------------------
        # SLICING
        # --------------------------------------------------------------

        "cpu_slicing_avg_pct":
            stage_resources["slicing"]["cpu_avg_pct"],

        "cpu_slicing_peak_pct":
            stage_resources["slicing"]["cpu_peak_pct"],

        "ram_slicing_avg_mb":
            stage_resources["slicing"]["ram_avg_mb"],

        "ram_slicing_peak_mb":
            stage_resources["slicing"]["ram_peak_mb"],

        # --------------------------------------------------------------
        # DETECTION
        # --------------------------------------------------------------

        "cpu_detection_avg_pct":
            stage_resources["detection"]["cpu_avg_pct"],

        "cpu_detection_peak_pct":
            stage_resources["detection"]["cpu_peak_pct"],

        "ram_detection_avg_mb":
            stage_resources["detection"]["ram_avg_mb"],

        "ram_detection_peak_mb":
            stage_resources["detection"]["ram_peak_mb"],

        # --------------------------------------------------------------
        # DESCRIPTION
        # --------------------------------------------------------------

        "cpu_description_avg_pct":
            stage_resources["description"]["cpu_avg_pct"],

        "cpu_description_peak_pct":
            stage_resources["description"]["cpu_peak_pct"],

        "ram_description_avg_mb":
            stage_resources["description"]["ram_avg_mb"],

        "ram_description_peak_mb":
            stage_resources["description"]["ram_peak_mb"],

        # --------------------------------------------------------------
        # OVERALL PROCESS TREE
        # --------------------------------------------------------------

        "cpu_avg_pct":
            overall_resources["cpu_avg_pct"],

        "cpu_peak_pct":
            overall_resources["cpu_peak_pct"],

        "ram_avg_mb":
            overall_resources["ram_avg_mb"],

        "ram_peak_mb":
            overall_resources["ram_peak_mb"],

        # --------------------------------------------------------------
        # DOCKER CONTAINER
        # --------------------------------------------------------------

        "container_ram_avg_mb":
            overall_resources["container_ram_avg_mb"],

        "container_ram_peak_mb":
            overall_resources["container_ram_peak_mb"],

        # --------------------------------------------------------------
        # PIPELINE METRICS
        # --------------------------------------------------------------

        "realtime_factor": round(
            realtime_factor,
            3
        ),

        "video_sec_per_processing_sec": round(
            video_sec_per_processing_sec,
            3
        ),

        "trucks_found": trucks_found,
    }

    # ==================================================================
    # CLEANUP
    # ==================================================================

    if not keep_outputs:
        storage.delete_job(
            job_id
        )

    return result


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video_dir",
        required=True,
    )

    parser.add_argument(
        "--keep_outputs",
        action="store_true",
    )

    parser.add_argument(
        "--precision",
        choices=[
            "FP32",
            "FP16",
            "INT8",
        ],
        default="INT8",
    )

    parser.add_argument(
        "--hint",
        choices=[
            "LATENCY",
            "THROUGHPUT",
        ],
        default="LATENCY",
    )

    parser.add_argument(
        "--out_csv",
        default=None,
    )

    args = parser.parse_args()

    if args.out_csv is None:
        args.out_csv = (
            f"benchmark_results_"
            f"{args.precision}_"
            f"{args.hint}.csv"
        )

    # If the output contains a directory, make sure it exists.
    output_dir = os.path.dirname(
        args.out_csv
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    video_extensions = (
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
    )

    video_files = sorted(
        filename
        for filename in os.listdir(
            args.video_dir
        )
        if filename.lower().endswith(
            video_extensions
        )
    )

    if not video_files:
        print(
            f"No videos found in "
            f"{args.video_dir}"
        )

        return

    print()
    print(
        f"Found {len(video_files)} videos."
    )

    print(
        f"Precision: {args.precision}"
    )

    print(
        f"Hint: {args.hint}"
    )

    print(
        f"Output: {args.out_csv}"
    )

    print()

    all_results = []

    for idx, video_name in enumerate(
        video_files,
        1,
    ):
        print(
            f"[{idx}/{len(video_files)}] "
            f"Processing: {video_name}"
        )

        video_path = os.path.join(
            args.video_dir,
            video_name,
        )

        result = run_one_video(
            video_path,
            keep_outputs=args.keep_outputs,
            precision=args.precision,
            hint=args.hint,
        )

        if result:
            all_results.append(
                result
            )

            print(
                f"  ✅ Total: "
                f"{result['time_total_sec']}s"
                f" | RTF: "
                f"{result['realtime_factor']}"
                f" | CPU avg: "
                f"{result['cpu_avg_pct']}%"
                f" | RAM peak: "
                f"{result['ram_peak_mb']}MB"
                f" | Container peak: "
                f"{result['container_ram_peak_mb']}"
                f"MB"
                f" | Trucks: "
                f"{result['trucks_found']}"
            )

            print()

    # ==================================================================
    # WRITE CSV
    # ==================================================================

    if all_results:

        with open(
            args.out_csv,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    all_results[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(
                all_results
            )

        print()
        print(
            f"📊 Saved benchmark table to "
            f"{args.out_csv}"
        )

    else:
        print()
        print(
            "No videos completed successfully."
        )


if __name__ == "__main__":
    main()