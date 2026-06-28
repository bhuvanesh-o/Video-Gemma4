# -*- coding: utf-8 -*-
"""
main.py

Top-level orchestrator. Runs the full pipeline in order:
  1. Gemma_Segment        — change-point detection, plots, segment timestamps → Excel
  2. Gemma_Slice_and_Excel_timestamps — cosine segmentation + physical video slicing → Excel
"""

import step_1_segmentation
import step_2_video_slice_excel_timestamp
import step_3_yoloXs_images
import step_4_prompt_document


def main():
    print("=" * 60)
    print("STEP 1: SEGMENT DETECTION & PLOTTING")
    print("=" * 60)
    step_1_segmentation.main()

    print("\n" + "=" * 60)
    print("STEP 2: VIDEO SLICING & EXCEL LOGGING")
    print("=" * 60)
    step_2_video_slice_excel_timestamp.main()

    print("\n" + "=" * 60)
    print("STEP 2: VIDEO SLICING & EXCEL LOGGING")
    print("=" * 60)
    step_3_yoloXs_images.main()

    print("\n" + "=" * 60)
    print("STEP 2: VIDEO SLICING & EXCEL LOGGING")
    print("=" * 60)
    step_4_prompt_document.main()

    print("\n🏁 Full pipeline complete.")


if __name__ == "__main__":
    main()