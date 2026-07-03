# progress.py
"""
Central progress-emission helper, used by all four pipeline step modules
(and their _functions.py libraries) so the exact same print() statements
you already have can ALSO be streamed to a browser as structured events via
FastAPI's Server-Sent Events (SSE).

REFACTOR NOTE (structured events): Previously this stored flat strings and
the UI just dumped them into a terminal-style log box. Now each event is a
small dict — {type, stage, text, meta, timestamp} — so the frontend can
render a proper "current stage" narrative (headline + live description) plus
a stage trail (done / active / pending), instead of a scrolling text log.
The terminal still prints the exact same message as before — nothing about
local/CLI behavior changes.
"""
import datetime

# In-memory store: {job_id: [list of event dicts]}
# Fine for a single-process dev server. If you ever run multiple worker
# processes in production, this would need to move to something shared
# (e.g. Redis) — not a concern yet.
_job_logs = {}


def _append(job_id, event):
    """Internal helper: timestamp an event dict and append it to the job's log."""
    if job_id is None:
        return
    event["timestamp"] = datetime.datetime.now().strftime("%H:%M:%S")
    _job_logs.setdefault(job_id, []).append(event)


def emit(message, job_id=None, stage=None, ui_message=None, meta=None):
    """
    Use this EVERYWHERE instead of print() inside the pipeline step files.

    - Always prints `message` locally, so running steps directly from the
      command line (no web app involved) behaves exactly as it does today.
    - If a job_id is provided, also appends a structured "stage_update" event
      to that job's in-memory log, so the SSE endpoint can stream it to the
      browser as JSON instead of raw text.

    stage: which pipeline stage this belongs to — "preparing", "segmentation",
           "slicing", or "detection". The frontend uses this to know which
           part of the UI (headline / trail dot) an update applies to.
    ui_message: the browser-facing text. If omitted, falls back to `message`
                (same text in both places — the original behavior). Pass a
                short, human-readable string here to say something different
                to the UI than what gets printed to the terminal.
    meta: optional small dict of extra structured data for the frontend
          (e.g. {"truck_found": True}) that doesn't need its own text line.
    """
    print(message)
    if job_id is None:
        return

    text_for_ui = ui_message if ui_message is not None else message
    # Collapse embedded newlines (e.g. emit("\nFoo...")) into spaces — a raw
    # "\n" inside an SSE payload can break browser-side parsing, and it's not
    # needed anymore now that each event is its own JSON object anyway.
    clean_text = text_for_ui.replace("\n", " ").strip()

    if not clean_text and not meta:
        return  # nothing worth sending to the UI for this call

    _append(job_id, {
        "type": "stage_update",
        "stage": stage,
        "text": clean_text,
        "meta": meta,
    })


def emit_stage_complete(stage, job_id=None):
    """Marks a whole stage ('preparing' / 'segmentation' / 'slicing' / 'detection') as finished."""
    print(f"--- Stage complete: {stage} ---")
    _append(job_id, {"type": "stage_complete", "stage": stage})


def emit_done(job_id=None):
    """Marks the entire job as successfully finished."""
    print("DONE")
    _append(job_id, {"type": "done"})


def emit_error(message, job_id=None):
    """Marks the entire job as failed, with a user-facing error message."""
    print(f"ERROR: {message}")
    _append(job_id, {"type": "error", "text": message})


def get_log(job_id):
    """Return everything emitted so far for a given job (used by the SSE endpoint)."""
    return _job_logs.get(job_id, [])


def clear_log(job_id):
    """Optional cleanup — call once a job's results have been delivered/downloaded."""
    _job_logs.pop(job_id, None)