# -*- coding: utf-8 -*-
"""
step_4_functions_prompt_document.py

REWRITTEN AGAIN: ported from a working Colab notebook that calls Gemma 4
26B A4B (free tier) via OpenRouter, using the openai SDK pointed at
OpenRouter's base_url. No torch/transformers/PIL needed anymore — this
whole stage is now just an HTTPS call + JSON parsing.

Retry logic, rate-limit backoff, and JSON-validation-before-returning are
carried over unchanged from the Colab version — that hardening is worth
keeping, since free-tier model output can be flaky.
"""
import os
import base64
import json
import re
import time
from openai import OpenAI

MODEL_ID = "google/gemma-4-26b-a4b-it:free"


def get_client(api_key):
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_json(raw_text):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output.")
    return json.loads(match.group(0))


def call_gemma_api(client, image_path, prompt, retries=3):
    """
    Unchanged logic from the Colab version: encodes the image, sends it with
    the structured-output prompt, validates the response is parseable JSON
    BEFORE returning (retrying on bad JSON), and handles rate-limit
    backoff separately from other errors. Also falls back to a plain call
    without extra_body if the reasoning-disable param isn't supported by
    whatever backend OpenRouter routes this particular request to.
    """
    b64_image = encode_image_base64(image_path)
    ext = os.path.splitext(image_path)[1].lstrip(".").lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64_image}"}}
        ]
    }]

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                temperature=0.1,
                max_tokens=1500,
                response_format={"type": "json_object"},
                extra_body={"reasoning": {"enabled": False}},
            )
            text = response.choices[0].message.content
            try:
                return extract_json(text)  # validated before returning
            except (ValueError, json.JSONDecodeError):
                last_error = "unparseable JSON in response"
                time.sleep(2)
                continue

        except Exception as e:
            err_str = str(e)
            last_error = err_str
            if "429" in err_str or "rate" in err_str.lower():
                time.sleep(5 * attempt)
                continue
            try:
                response = client.chat.completions.create(
                    model=MODEL_ID, messages=messages, temperature=0.1, max_tokens=1500,
                )
                return extract_json(response.choices[0].message.content)
            except Exception as e2:
                last_error = str(e2)
                time.sleep(3)

    raise RuntimeError(f"Failed after {retries} attempts. Last error: {last_error}")


def format_summary(parsed):
    """
    Turns the 7-field JSON dict into one compact, readable string for the
    single "E2B Analysis" Excel column — keeps the structured API response,
    but presents it as one cell rather than 7 separate columns.
    """
    parts = []
    field_labels = [
        ("VEHICLE TYPE", "vehicle_type"), ("COLOR", "color"), ("NUMBER PLATE", "number_plate"),
        ("BRAND", "brand"), ("LOAD", "load"), ("CONDITION", "condition"), ("NOTES", "additional_details"),
    ]
    # The prompt asks for numbered keys loosely — be forgiving about exact
    # key names the model chooses, since free-tier JSON adherence varies.
    for label, _ in field_labels:
        for key in parsed.keys():
            if label.replace(" ", "").lower() in key.replace("_", "").replace(" ", "").lower():
                value = parsed[key]
                if value:
                    parts.append(f"{label}: {value}")
                break
    return " | ".join(parts) if parts else json.dumps(parsed, ensure_ascii=False)