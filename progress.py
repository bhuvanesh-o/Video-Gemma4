# progress.py
"""
Central progress-emission helper, used by all four pipeline step modules
(and their _functions.py libraries) so the exact same print() statements
you already have can ALSO be streamed to a browser later via FastAPI's
Server-Sent Events (SSE) — without changing any of the surrounding logic.

Right now (no FastAPI wired up yet), this just behaves like print(), plus
it quietly stores each message in an in-memory per-job log. Once FastAPI
is added, the SSE endpoint will simply read from get_log(job_id) (or we'll
upgrade this to a queue for true real-time push — see note at bottom).
"""

import datetime

# In-memory store: {job_id: [list of message strings]}
# Fine for a single-process dev server. If you ever run multiple worker
# processes in production, this would need to move to something shared
# (e.g. Redis) — not a concern yet.
_job_logs = {}


def emit(message, job_id=None):
    """
    Use this EVERYWHERE instead of print() inside the pipeline step files.

    - Always prints locally, so running steps directly from the command
      line (no web app involved) behaves exactly as it does today.
    - If a job_id is provided, also appends the message (with a timestamp)
      to that job's in-memory log, so a later FastAPI endpoint can fetch
      and stream it to the browser.
    """
    print(message)
    if job_id is not None:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        _job_logs.setdefault(job_id, []).append(f"[{timestamp}] {message}")


def get_log(job_id):
    """Return everything emitted so far for a given job (used by the SSE endpoint later)."""
    return _job_logs.get(job_id, [])


def clear_log(job_id):
    """Optional cleanup — call once a job's results have been delivered/downloaded."""
    _job_logs.pop(job_id, None)


# NOTE FOR LATER (FastAPI step):
# get_log() returning the full list each time is fine for polling-based SSE
# (the endpoint re-reads the list every second and only sends new lines to
# the browser). If we want push-based streaming instead of polling, we'd
# swap _job_logs's list-per-job for an asyncio.Queue per job here — but
# that's a later optimization, not needed to get things working.