# -*- coding: utf-8 -*-
"""
main.py

Top-level CLI orchestrator — an ALTERNATIVE entrypoint to the FastAPI web app
(app.py / routers/pipeline.py). Runs the full pipeline in order, directly from
the command line, without ever starting a server or touching a browser:
  1. step_1_segmentation              — change-point detection, plots, segment timestamps → Excel
  2. step_2_video_slice_excel_timestamp — cosine segmentation + physical video slicing → Excel
  3. step_3_yoloXs_images             — YOLOX truck detection, tracking, crops → Excel
  4. step_4_prompt_document           — Gemma 4 multimodal inference → text report (optional, currently disabled below)

NOTE (known, unresolved as of this comment pass — left as-is intentionally):
step_1/2/3's main() functions all now require job_id as a parameter (part of
the storage.py job-folder refactor for the web app). The calls below
(step_1_segmentation.main(), etc.) don't pass job_id, so running this file
directly in its current form will raise a TypeError before segmentation even
starts. This file predates that refactor and hasn't been updated to match —
worth fixing (e.g. main(job_id="local_test")) before ever running main.py
directly; the web app path (uvicorn app:app) is unaffected by this, since
routers/pipeline.py always passes job_id correctly.

step_4_prompt_document's import and call are commented out below — step 4
currently only runs through the web app's routers/pipeline.py, not through
this CLI path.
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