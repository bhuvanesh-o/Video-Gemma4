# benchmark_videos.py
"""
Runs the full GEMMA pipeline (step_1 -> step_2 -> step_3) across a folder of
test videos, one at a time, and records timing + CPU/RAM usage for each — so
you can compare performance across different video lengths/resolutions/truck
counts instead of just getting one aggregate number.

Usage:
    python benchmark_videos.py --video_dir ./test_videos

Each video in --video_dir gets:
  1. Its own isolated job_id (so runs never share state or leftover files)
  2. The same upload path real users hit: storage.save_upload -> video_preprocess.validate_and_prepare
  3. Full step_1 -> step_2 -> step_3 run, timed per-stage
  4. A fresh ResourceMonitor sampling CPU%/RAM for just that video's run
  5. Job folder cleanup (storage.delete_job) before moving to the next video,
     so a 20-video benchmark run doesn't leave 20 jobs' worth of crops/videos
     sitting on disk afterward.

Results get written to benchmark_results.csv — one row per video.
"""
import os
import csv
import time
import uuid
import argparse

from resource_monitor import ResourceMonitor
import storage
import video_preprocess
import step_1_segmentation
import step_2_video_slice_excel_timestamp
import step_3_yoloXs_images


def count_trucks_found(job_id):
    """Reads the final registry Excel for this job and returns how many trucks were logged."""
    import pandas as pd
    excel_path = storage.path_for(job_id, "final_assets", "Vehicle_Registry_Master.xlsx")
    if not os.path.exists(excel_path):
        return 0
    df = pd.read_excel(excel_path)
    return len(df)


def run_one_video(video_path, keep_outputs=False, precision="INT8", hint="LATENCY"):
    """
    Runs the full pipeline on a single video file and returns a dict of
    metrics for that run. video_path is a local file OUTSIDE any job folder —
    this function copies it in via storage.save_upload, exactly like a real
    upload would.
    """
    video_name = os.path.basename(video_path)
    job_id = f"bench_{uuid.uuid4().hex[:8]}"

    # ── Simulate the real upload path ──────────────────────────────────────
    with open(video_path, "rb") as f:
        saved_path = storage.save_upload(job_id, f, filename=video_name)

    # Grab raw video metadata BEFORE any downscaling, so the results table
    # shows what the source file actually looked like.
    raw_info = video_preprocess.get_video_info(saved_path)

    try:
        video_preprocess.validate_and_prepare(saved_path, job_id=job_id)
    except ValueError as e:
        print(f"  ⚠️ Skipped {video_name}: {e}")
        storage.delete_job(job_id)
        return None

    # ── Run the pipeline with per-stage timing + resource monitoring ───────
    monitor = ResourceMonitor(interval=0.5)
    monitor.start()

    stage_times = {}
    overall_start = time.time()

    t0 = time.time()
    step_1_segmentation.main(job_id=job_id)
    stage_times["segmentation"] = time.time() - t0

    t0 = time.time()
    step_2_video_slice_excel_timestamp.main(job_id=job_id)
    stage_times["slicing"] = time.time() - t0

    t0 = time.time()
    step_3_yoloXs_images.main(job_id=job_id, model_precision=precision, performance_hint=hint)
    stage_times["detection"] = time.time() - t0

    total_time = time.time() - overall_start
    monitor.stop()

    trucks_found = count_trucks_found(job_id)

    result = {
        "video_name": video_name,
        "model_precision": precision,
        "performance_hint": hint,
        "duration_sec": round(raw_info["duration"], 2),
        "resolution": f"{raw_info['width']}x{raw_info['height']}",
        "time_segmentation_sec": round(stage_times["segmentation"], 2),
        "time_slicing_sec": round(stage_times["slicing"], 2),
        "time_detection_sec": round(stage_times["detection"], 2),
        "time_total_sec": round(total_time, 2),
        "cpu_avg_pct": round(sum(s[1] for s in monitor.samples) / len(monitor.samples), 1) if monitor.samples else 0,
        "cpu_peak_pct": round(max((s[1] for s in monitor.samples), default=0), 1),
        "ram_avg_mb": round(sum(s[2] for s in monitor.samples) / len(monitor.samples), 1) if monitor.samples else 0,
        "ram_peak_mb": round(max((s[2] for s in monitor.samples), default=0), 1),
        "trucks_found": trucks_found,
    }

    if not keep_outputs:
        storage.delete_job(job_id)  # wipe crops/videos/excel for this job — keeps disk clean across a big benchmark batch

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True, help="Folder containing test videos")
    parser.add_argument("--keep_outputs", action="store_true", help="Don't delete job folders after each run")
    parser.add_argument("--precision", choices=["FP32", "FP16", "INT8"], default="INT8")
    parser.add_argument("--hint", choices=["LATENCY", "THROUGHPUT"], default="LATENCY")
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()
    if args.out_csv is None:
        args.out_csv = f"benchmark_results_{args.precision}.csv"   # auto-name so runs never overwrite each other

    video_extensions = (".mp4", ".avi", ".mov", ".mkv")
    video_files = sorted(
        f for f in os.listdir(args.video_dir) if f.lower().endswith(video_extensions)
    )

    if not video_files:
        print(f"No videos found in {args.video_dir}")
        return

    print(f"Found {len(video_files)} videos. Starting benchmark...\n")

    all_results = []
    for idx, video_name in enumerate(video_files, 1):
        print(f"[{idx}/{len(video_files)}] Processing: {video_name}")
        video_path = os.path.join(args.video_dir, video_name)
        result = run_one_video(video_path, keep_outputs=args.keep_outputs, precision=args.precision, hint=args.hint)
        if result:
            all_results.append(result)
            print(f"  ✅ Total: {result['time_total_sec']}s | "
                  f"CPU avg: {result['cpu_avg_pct']}% | "
                  f"RAM peak: {result['ram_peak_mb']}MB | "
                  f"Trucks: {result['trucks_found']}\n")

    # ── Write comparison CSV ────────────────────────────────────────────────
    if all_results:
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n📊 Saved comparison table to {args.out_csv}")
    else:
        print("\nNo videos completed successfully.")


if __name__ == "__main__":
    main()


