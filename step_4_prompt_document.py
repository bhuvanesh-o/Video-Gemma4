# -*- coding: utf-8 -*-
"""
step_4_prompt_document.py

Runs Gemma 4 26B A4B (via OpenRouter, free tier) on the single best truck
crop step_3 saved for each logged vehicle, writing a structured summary
into a new "E2B Analysis" column on Vehicle_Registry_Master.xlsx.
"""
import os
import time
import pandas as pd


from openpyxl import load_workbook


import os
import requests

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY is not set. "
        "Set it in PowerShell before starting the pipeline."
    )










'''
So the headers are basically telling OpenRouter:

"Authorization" → this request is coming from your account/API key
"Content-Type": "application/json" → the data you're sending is JSON
'''

def analyze_image_with_openrouter(payload):

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload
    )

    response.raise_for_status()

    return response.json()







print("✅ OpenRouter API key loaded")

from step_4_functions_prompt_document import get_client, call_gemma_api, format_summary
from storage import dir_for
from progress import emit

CORE_PROMPT = (
    "You are analyzing one image of a truck captured by a traffic/CCTV camera. "
    "Look closely and answer the following parameters exactly, in this structured JSON format:\n\n"
    "1. VEHICLE TYPE: Specify the exact type of truck (e.g., container truck, tanker, flatbed, "
    "mini-truck, pickup, trailer, dump truck, etc.). If not a truck, state what it actually is.\n"
    "2. COLOR: State the primary body color, and secondary color if visible (e.g., cabin vs container).\n"
    "3. NUMBER PLATE: Read and transcribe the license/number plate text exactly as visible. "
    "If partially visible or unreadable, state 'partially visible: <what you can read>' or 'not readable'. "
    "Do not guess characters you cannot see clearly.\n"
    "4. BRAND / MAKE: If any manufacturer logo, brand name, or markings are visible (e.g., Tata, Ashok Leyland, "
    "Volvo, Eicher), state it. Otherwise say 'not visible'.\n"
    "5. LOAD / CARGO: Describe what the truck appears to be carrying, if anything is visible.\n"
    "6. CONDITION/DAMAGE: Note any visible damage, dents, rust, or unusual markings.\n"
    "7. ADDITIONAL DETAILS: Any other identifying features — stickers, text on the body, company name, etc.\n\n"
    "Be precise and objective. If a detail is not visible, explicitly say so instead of guessing."
)


def main(job_id):
    ASSET_DIR = dir_for(job_id, "final_assets")
    TRUCK_CROP_DIR = os.path.join(ASSET_DIR, "truck_crops")
    EXCEL_PATH = os.path.join(ASSET_DIR, "Vehicle_Registry_Master.xlsx")

    if not os.path.exists(EXCEL_PATH):
        emit("No registry found — skipping AI descriptions.", job_id=job_id, stage="describing", ui_message="")
        return

    df = pd.read_excel(EXCEL_PATH)
    if df.empty:
        emit("Registry is empty — skipping AI descriptions.", job_id=job_id, stage="describing", ui_message="")
        return

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        emit("⚠️ OPENROUTER_API_KEY not set — skipping AI descriptions.", job_id=job_id,
             stage="describing", ui_message="Skipping AI descriptions.")
        return

    client = get_client(api_key)
    descriptions = []
    total = len(df)

    for idx, row in df.iterrows():
        vehicle_id = str(row["Vehicle ID"])
        truck_img_path = os.path.join(TRUCK_CROP_DIR, f"{vehicle_id}_truck.jpg")

        if not os.path.exists(truck_img_path):
            descriptions.append("No image available")
            continue

        emit(f"🖼️ Analyzing {vehicle_id} ({idx + 1}/{total})...", job_id=job_id,
             stage="describing", ui_message=f"Analyzing {vehicle_id} ({idx + 1}/{total})...",
             meta={"clip_progress": {"current": idx + 1, "total": total}})

        try:
            parsed = call_gemma_api(client, truck_img_path, CORE_PROMPT)
            #desc = json.dumps(parsed, ensure_ascii=False, indent=2)  # raw JSON content, not a reformatted summary
            desc = format_summary(parsed)
        except Exception as e:
            desc = f"[analysis failed: {e}]"
            emit(f"⚠️ Analysis failed for {vehicle_id}: {e}", job_id=job_id, stage="describing", ui_message="")

        descriptions.append(desc)
        time.sleep(1)  # gentle on the free-tier rate limit, same as the Colab version
    
    #df["E2B Analysis"] = descriptions
    #df.to_excel(EXCEL_PATH, index=False)
    
    # --------------------------------------------------------------------------
    # WRITE ONLY THE E2B ANALYSIS COLUMN
    # Preserve all existing Excel formulas/hyperlinks
    # --------------------------------------------------------------------------

    wb = load_workbook(EXCEL_PATH, data_only=False)
    ws = wb.active


    # Find existing column headers
    headers = {
        cell.value: cell.column
        for cell in ws[1]
    }


    # If E2B Analysis already exists, reuse it.
    # Otherwise create a new column.
    if "E2B Analysis" in headers:
        analysis_col = headers["E2B Analysis"]

    else:
        analysis_col = ws.max_column + 1
        ws.cell(
            row=1,
            column=analysis_col,
            value="E2B Analysis"
        )


    # Write each description.
    # Excel row 1 = headers, so data begins at row 2.
    for idx, desc in enumerate(descriptions, start=2):

        ws.cell(
            row=idx,
            column=analysis_col,
            value=desc
        )


    # Save without destroying the existing hyperlinks/formulas
    wb.save(EXCEL_PATH)
    
    emit(f"✅ AI descriptions added for {total} vehicles.", job_id=job_id,
         stage="describing", ui_message="Descriptions added to your registry.")


if __name__ == "__main__":
    main(job_id="local_test")