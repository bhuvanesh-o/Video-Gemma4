# -*- coding: utf-8 -*-
"""
main.py

Top-level orchestrator. Runs the full pipeline in order:
  1. step_1_segmentation              — change-point detection, plots, segment timestamps → Excel
  2. step_2_video_slice_excel_timestamp — cosine segmentation + physical video slicing → Excel
  3. step_3_yoloXs_images             — YOLOX truck detection, tracking, crops → Excel
  4. step_4_prompt_document           — Gemma 4 multimodal inference → text report (optional)
"""

import step_1_segmentation
import step_2_video_slice_excel_timestamp
import step_3_yoloXs_images
# import step_4_prompt_document


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
    print("STEP 3: VIDEO SLICING & EXCEL LOGGING")
    print("=" * 60)
    step_3_yoloXs_images.main()

    '''
    print("\n" + "=" * 60)
    print("STEP 4: VIDEO SLICING & EXCEL LOGGING")
    print("=" * 60)
    step_4_prompt_document.main()
    '''

    print("\n🏁 Full pipeline complete.")


if __name__ == "__main__":
    main()