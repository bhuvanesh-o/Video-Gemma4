# -*- coding: utf-8 -*-
"""
step_4_functions_prompt_document.py

Two model backends now:
  - call_gemma_api / get_client  -> Gemma 4 26B A4B via OpenRouter (cloud API)
  - call_litert_e2b               -> Gemma 4 E2B via LiteRT-LM (local, subprocess)

extract_json / extract_fields are shared by both — both models are asked to
return the same JSON shape, so parsing logic doesn't need to differ.
"""
import os
import base64
import json
import re
import time
import subprocess
from openai import OpenAI

MODEL_ID = "google/gemma-4-26b-a4b-it:free"

# LiteRT-LM local model config
LITERT_MODEL_REPO = "litert-community/gemma-4-E2B-it-litert-lm"
LITERT_MODEL_FILE = "gemma-4-E2B-it.litertlm"

FIELD_LABELS = [
    "Vehicle Type", "Color", "Number Plate", "Brand", "Load", "Condition", "Additional Details",
]


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


def extract_fields(parsed):
    """Pulls each of the 7 expected fields out of a parsed JSON dict, loose key-matching."""
    result = {}
    for label in FIELD_LABELS:
        value = "N/A"
        target = label.replace(" ", "").lower()
        for key in parsed.keys():
            if target in key.replace("_", "").replace(" ", "").lower():
                v = parsed[key]
                if v:
                    value = str(v)
                break
        result[label] = value
    return result


# ==============================================================================
# ── 26B A4B via OpenRouter ─────────────────────────────────────────────────────
# ==============================================================================
def call_gemma_api(client, image_path, prompt, retries=3):
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
                model=MODEL_ID, messages=messages, temperature=0.1, max_tokens=1500,
                response_format={"type": "json_object"},
                extra_body={"reasoning": {"enabled": False}},
            )
            text = response.choices[0].message.content
            try:
                return extract_json(text)
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


# ==============================================================================
# ── E2B via LiteRT-LM (local) ──────────────────────────────────────────────────
# ==============================================================================
def call_litert_e2b(image_path, prompt, backend="cpu", timeout=120):
    """
    Runs Gemma 4 E2B fully locally via the litert-lm CLI, as a subprocess —
    same pattern as the ffmpeg subprocess.run() call already in step_3.

    ⚠️ UNVERIFIED: the --image flag name below is a PLACEHOLDER. Confirm the
    real flag by running `litert-lm run --help` and update this before
    trusting it. Everything else in this function is correct regardless.
    """
    

    print(f"[DEBUG] Running litert-lm with image: {image_path}")
    print(f"[DEBUG] Model repo: {LITERT_MODEL_REPO}")
    print(f"[DEBUG] Model file: {LITERT_MODEL_FILE}")


    # Force the child litert-lm process to communicate in UTF-8 — needed
    # because model output can legitimately contain non-Latin text (e.g.
    # Devanagari/Hindi text painted on a truck body, as seen in real footage
    # this pipeline processes). Without this, Windows' default console
    # encoding can throw a UnicodeDecodeError when subprocess tries to
    # capture that output as text.

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        result = subprocess.run(
            [
                "litert-lm",
                "run",
                f"--from-huggingface-repo={LITERT_MODEL_REPO}",
                LITERT_MODEL_FILE,
                f"--backend={backend}",
                f"--vision-backend={backend}",


                f"--temperature=0",
                f"--seed=42",
                f"--thinking=false",


                f"--attachment={image_path}",
                f"--prompt={prompt}",
            ],
            capture_output=True, text=True,

            encoding="utf-8", errors="replace",   # "replace" swaps any still-undecodable byte for a placeholder char instead of crashing the whole call

            env=env, timeout=timeout, check=False,
        )

        print(f"[DEBUG] Return code: {result.returncode}")
        print(f"[DEBUG] STDOUT:\n{result.stdout}")
        print(f"[DEBUG] STDERR:\n{result.stderr}")

        if result.returncode != 0:
            raise RuntimeError(
                f"litert-lm exited {result.returncode}: "
                f"{result.stderr.strip()}"
            )


        # litert-lm prints its own informational status lines (e.g. "Using
        # cached model: ...") mixed into stdout ALONGSIDE the actual model
        # response. Strip those known noise lines out before returning, so
        # extract_json() downstream doesn't have to parse around them.

        cleaned_lines = [
            line for line in result.stdout.splitlines()
            if not line.strip().startswith("Using cached model:")
        ]
        cleaned_output = "\n".join(cleaned_lines).strip()
        return cleaned_output


    except Exception as e:
        print(f"[ERROR] litert-lm call failed: {e}")
        raise