# benchmark_videos.py
"""
Runs the full GEMMA pipeline (preprocess -> step_1 -> step_2 -> step_3 -> step_4)
across a folder of test videos, recording per-stage timing and CPU/RAM usage.

NOTE: per-model (26B vs E2B) timing/success-rate breakdown is NOT included
here — that requires step_4_prompt_document.main() to actually build and
return a stats dict, which it doesn't do today. See the note at the bottom
of this file for what that would take.
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


def summarize_monitor(monitor):
    if not monitor.samples:
        return {"cpu_avg_pct": 0, "cpu_peak_pct": 0, "ram_avg_mb": 0, "ram_peak_mb": 0}
    return {
        "cpu_avg_pct": round(sum(s[1] for s in monitor.samples) / len(monitor.samples), 1),
        "cpu_peak_pct": round(max(s[1] for s in monitor.samples), 1),
        "ram_avg_mb": round(sum(s[2] for s in monitor.samples) / len(monitor.samples), 1),
        "ram_peak_mb": round(max(s[2] for s in monitor.samples), 1),
    }


def count_trucks_found(job_id):
    import pandas as pd
    excel_path = storage.path_for(job_id, "final_assets", "Vehicle_Registry_Master.xlsx")
    if not os.path.exists(excel_path):
        return 0
    return len(pd.read_excel(excel_path))


def run_one_video(video_path, keep_outputs=False, precision="INT8", hint="LATENCY"):
    video_name = os.path.basename(video_path)
    job_id = f"bench_{uuid.uuid4().hex[:8]}"
    stage_times = {}

    monitor = ResourceMonitor(interval=0.5)
    monitor.start()
    overall_start = time.perf_counter()

    with open(video_path, "rb") as f:
        saved_path = storage.save_upload(job_id, f, filename=video_name)
    raw_info = video_preprocess.get_video_info(saved_path)

    # ── Preprocessing ────────────────────────────────────────────────────
    preprocess_monitor = ResourceMonitor(interval=0.2)
    preprocess_monitor.start()
    t0 = time.perf_counter()
    try:
        video_preprocess.validate_and_prepare(saved_path, job_id=job_id)
    except ValueError as e:
        print(f"  ⚠️ Skipped {video_name}: {e}")
        preprocess_monitor.stop()
        monitor.stop()
        if not keep_outputs:
            storage.delete_job(job_id)
        return None
    stage_times["preprocess"] = time.perf_counter() - t0
    preprocess_monitor.stop()
    preprocess_resources = summarize_monitor(preprocess_monitor)

    # ── Step 1 ────────────────────────────────────────────────────────────
    segmentation_monitor = ResourceMonitor(interval=0.2)
    segmentation_monitor.start()
    t0 = time.perf_counter()
    step_1_segmentation.main(job_id=job_id)
    stage_times["segmentation"] = time.perf_counter() - t0
    segmentation_monitor.stop()
    segmentation_resources = summarize_monitor(segmentation_monitor)

    # ── Step 2 ────────────────────────────────────────────────────────────
    slicing_monitor = ResourceMonitor(interval=0.2)
    slicing_monitor.start()
    t0 = time.perf_counter()
    step_2_video_slice_excel_timestamp.main(job_id=job_id)
    stage_times["slicing"] = time.perf_counter() - t0
    slicing_monitor.stop()
    slicing_resources = summarize_monitor(slicing_monitor)

    # ── Step 3 ────────────────────────────────────────────────────────────
    # NOTE: matches step_3_yoloXs_images.main()'s ACTUAL signature —
    # no collect_benchmark param, no return value. Any YOLOX-vs-PaddleOCR
    # split would need real instrumentation added inside step_3 first.
    detection_monitor = ResourceMonitor(interval=0.2)
    detection_monitor.start()
    t0 = time.perf_counter()
    step_3_yoloXs_images.main(job_id=job_id, model_precision=precision, performance_hint=hint)
    stage_times["detection"] = time.perf_counter() - t0
    detection_monitor.stop()
    detection_resources = summarize_monitor(detection_monitor)

    trucks_found = count_trucks_found(job_id)

    # ── Step 4 ────────────────────────────────────────────────────────────
    # NOTE: step_4_prompt_document.main() returns None today — description_monitor
    # below (which recursively includes the litert-lm subprocess's CPU/RAM,
    # via ResourceMonitor's existing children() tracking) is your real signal
    # for Step 4's resource cost, not a stats dict from step_4 itself.
    description_monitor = ResourceMonitor(interval=0.2)
    description_monitor.start()
    t0 = time.perf_counter()
    step_4_prompt_document.main(job_id=job_id)
    stage_times["description"] = time.perf_counter() - t0
    description_monitor.stop()
    description_resources = summarize_monitor(description_monitor)

    total_time = time.perf_counter() - overall_start
    monitor.stop()
    overall_resources = summarize_monitor(monitor)

    duration = raw_info["duration"]
    realtime_factor = total_time / duration if duration > 0 else 0
    video_sec_per_processing_sec = duration / total_time if total_time > 0 else 0

    result = {
        "video_name": video_name,
        "model_precision": precision,
        "performance_hint": hint,
        "duration_sec": round(duration, 2),
        "resolution": f"{raw_info['width']}x{raw_info['height']}",

        "time_preprocess_sec": round(stage_times["preprocess"], 2),
        "time_segmentation_sec": round(stage_times["segmentation"], 2),
        "time_slicing_sec": round(stage_times["slicing"], 2),
        "time_detection_sec": round(stage_times["detection"], 2),
        "time_description_sec": round(stage_times["description"], 2),
        "time_total_sec": round(total_time, 2),

        "cpu_preprocess_avg_pct": preprocess_resources["cpu_avg_pct"],
        "cpu_preprocess_peak_pct": preprocess_resources["cpu_peak_pct"],
        "ram_preprocess_avg_mb": preprocess_resources["ram_avg_mb"],
        "ram_preprocess_peak_mb": preprocess_resources["ram_peak_mb"],

        "cpu_segmentation_avg_pct": segmentation_resources["cpu_avg_pct"],
        "cpu_segmentation_peak_pct": segmentation_resources["cpu_peak_pct"],
        "ram_segmentation_avg_mb": segmentation_resources["ram_avg_mb"],
        "ram_segmentation_peak_mb": segmentation_resources["ram_peak_mb"],

        "cpu_slicing_avg_pct": slicing_resources["cpu_avg_pct"],
        "cpu_slicing_peak_pct": slicing_resources["cpu_peak_pct"],
        "ram_slicing_avg_mb": slicing_resources["ram_avg_mb"],
        "ram_slicing_peak_mb": slicing_resources["ram_peak_mb"],

        "cpu_detection_avg_pct": detection_resources["cpu_avg_pct"],
        "cpu_detection_peak_pct": detection_resources["cpu_peak_pct"],
        "ram_detection_avg_mb": detection_resources["ram_avg_mb"],
        "ram_detection_peak_mb": detection_resources["ram_peak_mb"],

        "cpu_description_avg_pct": description_resources["cpu_avg_pct"],
        "cpu_description_peak_pct": description_resources["cpu_peak_pct"],
        "ram_description_avg_mb": description_resources["ram_avg_mb"],
        "ram_description_peak_mb": description_resources["ram_peak_mb"],

        "cpu_avg_pct": overall_resources["cpu_avg_pct"],
        "cpu_peak_pct": overall_resources["cpu_peak_pct"],
        "ram_avg_mb": overall_resources["ram_avg_mb"],
        "ram_peak_mb": overall_resources["ram_peak_mb"],

        "realtime_factor": round(realtime_factor, 3),
        "video_sec_per_processing_sec": round(video_sec_per_processing_sec, 3),
        "trucks_found": trucks_found,
    }

    if not keep_outputs:
        storage.delete_job(job_id)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--keep_outputs", action="store_true")
    parser.add_argument("--precision", choices=["FP32", "FP16", "INT8"], default="INT8")
    parser.add_argument("--hint", choices=["LATENCY", "THROUGHPUT"], default="LATENCY")
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    if args.out_csv is None:
        args.out_csv = f"benchmark_results_{args.precision}_{args.hint}.csv"

    video_extensions = (".mp4", ".avi", ".mov", ".mkv")
    video_files = sorted(f for f in os.listdir(args.video_dir) if f.lower().endswith(video_extensions))
    if not video_files:
        print(f"No videos found in {args.video_dir}")
        return

    print(f"Found {len(video_files)} videos.\nPrecision: {args.precision} | Hint: {args.hint} | Output: {args.out_csv}\n")

    all_results = []
    for idx, video_name in enumerate(video_files, 1):
        print(f"[{idx}/{len(video_files)}] Processing: {video_name}")
        video_path = os.path.join(args.video_dir, video_name)
        result = run_one_video(video_path, keep_outputs=args.keep_outputs, precision=args.precision, hint=args.hint)
        if result:
            all_results.append(result)
            print(f"  ✅ Total: {result['time_total_sec']}s | RTF: {result['realtime_factor']} | "
                  f"CPU avg: {result['cpu_avg_pct']}% | RAM peak: {result['ram_peak_mb']}MB | "
                  f"Trucks: {result['trucks_found']}\n")

    if all_results:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n📊 Saved benchmark table to {args.out_csv}")
    else:
        print("\nNo videos completed successfully.")


if __name__ == "__main__":
    main()