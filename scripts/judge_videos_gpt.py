#!/usr/bin/env python3
"""Judge generated videos with a GPT-5.x multimodal chat API against frozen
QA checklists -- single task or batch, the OpenAI-compatible analogue of
scripts/judge_videos_gemini.py.

Videos are read from ``data/outputs/<model-name>/<task-id>/video.mp4`` and the
matching checklist from ``data/checklists/<task-id>_qa.json`` (1:1 by task id).
Pass ``--model-name wan2.2`` to select a generator, a comma-separated subset
(``--model-name wan2.2,cosmos3-nano``), or ``--model-name all`` to sweep every
generator under data/outputs -- each model is a section of the same run, so one
command judges all models x all tasks.  Add ``--task-id`` to select tasks by
their globally-unique leading number (``001`` one, ``001-020`` an inclusive
range, ``001,003,005`` discrete), or by a full/partial dir name such as
``001_L1_001_agronomy``.  Without ``--task-id``, every task under each model dir
is judged in batch.  Task discovery, checklist loading, prompt rendering
(vlm_judge.txt paragraph 1 -> system message) and answer validation are
imported from scripts/judge_videos_gemini.py, so both judges stay in lockstep.

Video input.  chat/completions accepts images, not video, so each clip is
decoded locally with ffmpeg at ``--fps`` (default 4) keyframes and every frame is
attached as an ``image_url`` part carrying a base64 data URL
(``data:image/jpeg;base64,...``) in temporal order, followed by the judge text.
``--max-frames`` caps the payload: for a long clip the effective rate drops to
``max_frames / duration`` so the whole clip stays covered instead of being
truncated.  Frames are never written to disk; ffmpeg fills a temporary directory
that is removed right after base64 encoding.
``--image-format png`` switches the container to PNG (lossless, but a far larger
payload).  Note this is a *sampling* pass for the judge -- it deliberately does
not decode every frame, since clips such as ``079_L1_072_physics`` carry
duplicate frame timestamps and would inflate the payload for no gain.

The judge prompt is rendered from data/rules/vlm_judge.txt exactly as in
bench.eval.vlm_judge.build_judge_prompt: paragraph 1 becomes the system message,
the rest (embedded checklist + rules + output format) is the user prompt.  The
model's OUTPUT-FORMAT JSON answer is validated against the frozen checklist and
saved to

    <output-dir>/<judge-label>/<model-name>/<task-id>.json

with ``<output-dir>`` defaulting to data/results/fps<--fps> (the fps bucket
matches ``--fps``, so GPT answers sit next to the Gemini ones for the same frame
budget) and ``<judge-label>`` to the API model id (e.g.
data/results/fps8/<model>/wan2.2/001_L1_001_agronomy.json).

Credentials.  Ignored in --prompt-only/--dry-run; a real call authenticates with
exactly one bearer token, read from the environment (never stored in this file):
    GPT_API_TOKEN  -> sent as  Authorization: Bearer <token>
You may also pass --token explicitly.

Endpoint.  The base URL is not hardcoded: pass --base-url or set
JUDGE_API_BASE_URL.  Any OpenAI-compatible chat/completions endpoint works.

Requirements.  ffmpeg + ffprobe on PATH (frame sampling), ``requests``, and an
endpoint that accepts image_url parts.  A transport error or an answer that fails
checklist validation is retried ``--retries`` times with exponential backoff
(frames are decoded once per task, only the API call repeats); a task that still
fails prints ERROR and the batch continues, the same as judge_videos_gemini.py.
A full sweep is resumable (existing answers are skipped without --overwrite) and
``--workers N`` judges N tasks of the same model concurrently.

Examples:
    # Inspect the judge prompt for one task (no API call, no media read)
    python scripts/judge_videos_gpt.py --model-name wan2.2 --task-id 001 --prompt-only

    # Show the payload structure for one task (frames extracted, no API call)
    python scripts/judge_videos_gpt.py --model-name wan2.2 --task-id 001 --dry-run

    # Judge a single task, a range, or several discrete numbers of wan2.2
    python scripts/judge_videos_gpt.py --model-name wan2.2 --task-id 001
    python scripts/judge_videos_gpt.py --model-name wan2.2 --task-id 001-020
    python scripts/judge_videos_gpt.py --model-name wan2.2 --task-id 001,003,005

    # Judge every task under data/outputs/wan2.2 in batch
    python scripts/judge_videos_gpt.py --model-name wan2.2

    # Judge ALL tasks of ALL generators (one sweep; resumable, see below)
    python scripts/judge_videos_gpt.py --model-name all

    # Same sweep, but only the first 3 tasks of each model (smoke test); a
    # subset of generators instead of all of them
    python scripts/judge_videos_gpt.py --model-name all --limit 3
    python scripts/judge_videos_gpt.py --model-name wan2.2,cosmos3-super

    # Full sweep in parallel, 6 tasks at a time, detached from the terminal
    nohup python scripts/judge_videos_gpt.py --model-name all --workers 6 > sweep.log 2>&1 &
    tail -f sweep.log          # live progress, one block per finished task

    # Batch but cap to the first 3 (smoke test); re-judge instead of skipping
    python scripts/judge_videos_gpt.py --model-name wan2.2 --limit 3
    python scripts/judge_videos_gpt.py --model-name wan2.2 --overwrite

    # Cheaper/faster sweep: 2 fps and a shorter reasoning budget
    python scripts/judge_videos_gpt.py --model-name wan2.2 --fps 2 --reasoning-effort low
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from typing import Any, Callable

import requests

# Make `bench` and `scripts` importable when launched as a plain script.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bench.eval.qa import _parse_response  # noqa: E402
from bench.eval.vlm_judge import (  # noqa: E402
    QAItem,
    QAJudgement,
    _validate_judge_payload,
    build_judge_prompt,
    judge_payload,
)
# Shared with the Gemini judge: task discovery, checklist loading, prompt
# partitioning (paragraph 1 -> system message) and result-path layout.
from scripts.judge_videos_gemini import (  # noqa: E402
    DEFAULT_RULES,
    VIDEO_ROOT,
    _result_path,
    collect_tasks,
    load_checklist,
    parse_task_selection,
    partition_messages,
)

# ---------------------------------------------------------------------------
# Path / API defaults
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_FPS = 4               # keyframes sampled per second of generated video
DEFAULT_MAX_FRAMES = 128       # cap; long clips get a lower effective fps instead
DEFAULT_JPEG_QUALITY = 3      # ffmpeg -q:v for mjpeg (1 best ... 31 worst)

# Frame container.  jpeg keeps an L2 clip at ~2 MB of base64; png is lossless but
# ~20x bigger (see --image-format).
_IMAGE_SUFFIX = {"jpeg": ".jpg", "png": ".png"}
_IMAGE_MIME = {"jpeg": "image/jpeg", "png": "image/png"}

# No API endpoint is hardcoded: pass --base-url or export JUDGE_API_BASE_URL.
# Any OpenAI-compatible chat/completions endpoint works.
DEFAULT_BASE_URL = os.environ.get("JUDGE_API_BASE_URL", "")

# A reasoning judge over ~50 base64 frames occasionally trips a transient
# transport error (connection reset / network drop / HTTP 5xx) or answers with a
# payload that fails checklist validation, which must not cost a whole batch run.
DEFAULT_RETRIES = 2           # extra attempts after the first one (0 = off)
RETRY_BACKOFF_S = 5.0         # doubled per attempt, capped at 60s

# No credential is stored in this file.  Export GPT_API_TOKEN, or pass --token.
# The answers bucket name follows --fps, so it is resolved in main(), not here.
def _answers_dir(fps: float) -> pathlib.Path:
    """Answers bucket for one sampling rate: data/results/fps<--fps>/, which is
    how the records shipped with this repository are laid out."""
    return DATA_DIR / "results" / f"fps{fps:g}"
# ---------------------------------------------------------------------------
# Video -> keyframes -> base64 (the chat endpoint has no video part)
# ---------------------------------------------------------------------------
def probe_duration(video_path: pathlib.Path) -> float:
    """Clip duration in seconds via ffprobe (raises when it cannot be read)."""
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found on PATH; install ffmpeg to sample frames")
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video_path}: {proc.stderr.strip()[:300]}")
    try:
        return float(proc.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"ffprobe returned no duration for {video_path}") from exc


def extract_frames(
    video_path: pathlib.Path,
    *,
    fps: float = DEFAULT_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
    quality: int = DEFAULT_JPEG_QUALITY,
    image_format: str = "jpeg",
) -> tuple[list[str], dict[str, Any]]:
    """Sample ``video_path`` at ``fps`` -> (base64 jpeg frames, sampling meta).

    One ffmpeg pass writes JPEG keyframes into a temporary directory that is
    removed before returning; frames keep their temporal order.  When the clip
    would yield more than ``max_frames`` frames, the sampling rate is lowered to
    ``max_frames / duration`` so the whole clip stays covered.  meta holds
    {duration, fps, n_frames, image_bytes, image_format} for logging/wording.

    The ``fps`` filter re-times the stream to a constant frame rate, so source
    clips with broken or duplicated frame timestamps (several generator outputs
    have equal ``best_effort_timestamp`` values, which makes ``-vsync 0`` fail the
    image2 muxer with "non monotonically increasing dts") still extract cleanly
    here -- no frame is rejected for a timestamp collision, and no frame is
    silently dropped either: the ffmpeg exit status and the resulting frame count
    are both checked against ``duration * fps``.

    This is a sampling pass for the judge (4 fps by default), not the 1:1
    every-frame export; that path keeps every
    decoded frame via ``setpts=N`` + ``-fps_mode passthrough`` and writes PNG.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; install ffmpeg to sample frames")
    if fps <= 0:
        raise ValueError(f"--fps must be positive, got {fps}")
    if image_format not in _IMAGE_SUFFIX:
        raise ValueError(f"unsupported --image-format {image_format!r}")

    duration = probe_duration(video_path)
    rate = float(fps)
    if max_frames and duration > 0:
        rate = min(rate, max_frames / duration)

    with tempfile.TemporaryDirectory(prefix="labinstruct_frames_") as tmp:
        tmp_dir = pathlib.Path(tmp)
        suffix = _IMAGE_SUFFIX[image_format]
        command = ["ffmpeg", "-y", "-v", "error", "-i", str(video_path),
                   "-vf", f"fps={rate:.6f}"]
        if image_format == "jpeg":
            command += ["-q:v", str(quality)]
        if max_frames:
            command += ["-frames:v", str(max_frames + 2)]
        command += [str(tmp_dir / ("frame_%04d" + suffix))]
        proc = subprocess.run(command, capture_output=True, text=True, timeout=600)
        paths = sorted(tmp_dir.glob(f"frame_*{suffix}"))
        if not paths:
            raise RuntimeError(
                f"ffmpeg extracted no frames from {video_path}: {proc.stderr.strip()[:300]}"
            )
        expected = int(duration * rate) if duration > 0 else len(paths)
        if proc.returncode != 0 or not expected <= len(paths) <= expected + 2:
            raise RuntimeError(
                f"ffmpeg extracted {len(paths)} frame(s) from {video_path}, expected "
                f"~{expected} at {rate:g} fps (exit {proc.returncode}): "
                f"{proc.stderr.strip()[:300]}"
            )
        if max_frames and len(paths) > max_frames:
            # Evenly subsample (never drop just the tail of the clip).
            if max_frames == 1:
                paths = [paths[0]]
            else:
                step = (len(paths) - 1) / (max_frames - 1)
                paths = [paths[round(i * step)] for i in range(max_frames)]
        blobs = [path.read_bytes() for path in paths]

    frames = [base64.b64encode(blob).decode("ascii") for blob in blobs]
    meta = {
        "duration": duration,
        "fps": round(len(frames) / duration, 3) if duration > 0 else rate,
        "n_frames": len(frames),
        "image_bytes": sum(len(blob) for blob in blobs),
        "image_format": image_format,
    }
    return frames, meta


def frames_media_text(n_frames: int | None, fps: float) -> str:
    """Text for the {{GENERATED_VIDEO}} placeholder of a keyframe feed.

    ``n_frames=None`` is used by --prompt-only, where no media is read yet, so the
    marker describes the frame feed in general instead of a concrete count.
    """
    rate = f"{fps:g}"
    if n_frames is None:
        return ("Provided as the keyframe images that follow, sampled from the "
                f"generated video at {rate} fps in temporal order.")
    if n_frames == 1:
        return ("Provided as the single image that follows, the first frame of the "
                "generated video.")
    return ("Provided as the "
            f"{n_frames} images that follow, sampled from the generated video at "
            f"{rate} fps in temporal order (image 1 is the start of the video).")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
def build_judge_text(
    spec: dict[str, Any],
    level: str,
    qas: list[QAItem],
    rules_path: pathlib.Path = DEFAULT_RULES,
    *,
    fps: float = DEFAULT_FPS,
    n_frames: int | None = None,
) -> str:
    """Render the frozen vlm_judge.txt rule + embedded checklist (no media bytes)."""
    return build_judge_prompt(
        spec, qas, level,
        video_text=frames_media_text(n_frames, fps), template_path=rules_path,
    )


# ---------------------------------------------------------------------------
# OpenAI-compatible chat payload
# ---------------------------------------------------------------------------
def image_url_part(
    frame_b64: str, detail: str | None = None, image_format: str = "jpeg",
) -> dict[str, Any]:
    """One base64 frame as an ``image_url`` chat part (data URL, no upload)."""
    mime = _IMAGE_MIME.get(image_format, "image/jpeg")
    image: dict[str, Any] = {"url": f"data:{mime};base64,{frame_b64}"}
    if detail:
        image["detail"] = detail
    return {"type": "image_url", "image_url": image}


def build_payload(
    *,
    model: str,
    user_text: str,
    frames: list[str],
    system_text: str | None = None,
    include_frames: bool = True,
    request_json: bool = True,
    detail: str | None = None,
    image_format: str = "jpeg",
    temperature: float | None = None,
    max_completion_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Build the chat/completions body (a dict; json.dumps happens at send time).

    Frames come first and the instruction text last, mirroring the Gemini judge
    (media first).  With ``include_frames=False`` the image parts collapse into one
    text placeholder so --dry-run shows the structure without megabytes of base64.

    ``request_json`` adds response_format={"type": "json_object"} so the model
    returns the OUTPUT-FORMAT object as raw JSON rather than Markdown/commentary
    (the analogue of Gemini's responseMimeType); disable it with --no-json-request
    on endpoints that reject the field -- the text rules still ask for JSON.
    ``temperature`` is left unset by default because gpt-5.x reasoning models
    ignore or reject values other than 1.
    """
    if include_frames and frames:
        media_parts: list[dict[str, Any]] = [
            image_url_part(frame, detail, image_format) for frame in frames
        ]
    else:
        mime = _IMAGE_MIME.get(image_format, "image/jpeg")
        note = (f"[{len(frames)} image_url parts omitted; base64 "
                f"{sum(len(frame) for frame in frames):,} chars, mime {mime}]"
                if frames else "[no frames]")
        media_parts = [{"type": "text", "text": note}]

    messages: list[dict[str, Any]] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user",
                     "content": media_parts + [{"type": "text", "text": user_text}]})

    body: dict[str, Any] = {"model": model, "messages": messages}
    if temperature is not None:
        body["temperature"] = temperature
    if max_completion_tokens:
        body["max_completion_tokens"] = int(max_completion_tokens)
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    if request_json:
        body["response_format"] = {"type": "json_object"}
    return body


def _read_stream(response: requests.Response, model: str = "") -> dict[str, Any]:
    """Collapse an OpenAI SSE stream into a chat.completion-shaped dict.

    Only ``data:`` JSON chunks are read; ``delta.content`` pieces are joined into
    one message, the last ``usage`` chunk is kept, and ``data: [DONE]`` ends it.
    Streaming keeps bytes moving while the model reasons, so a long thinking
    pause cannot look like a stalled connection.
    """
    content: list[str] = []
    usage: dict[str, Any] | None = None
    finish: str | None = None
    ended = False
    for raw in response.iter_lines():
        if not raw or not raw.startswith(b"data:"):
            continue
        payload = raw[5:].strip()
        if payload == b"[DONE]":
            ended = True
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                content.append(delta["content"])
            finish = choice.get("finish_reason") or finish
    text = "".join(content).strip()
    if not text or not (ended or finish):
        # The connection can drop mid-answer; the joined text is then truncated
        # JSON that would fail validation much later. Fail loudly so --retries
        # simply asks the API again.
        raise RuntimeError(
            f"API stream ended early (model={model}): {len(text)} chars of "
            f"content, finish_reason={finish!r}, [DONE]={ended}"
        )
    return {"choices": [{"message": {"content": text}, "finish_reason": finish}],
            "usage": usage}


def call_chat_completions(
    body: dict[str, Any],
    *,
    base_url: str,
    model: str,
    token: str | None,
    timeout_s: int = 600,
    stream: bool = False,
) -> dict[str, Any]:
    """POST the payload to <base_url>/chat/completions.

    Authenticates with ``Authorization: Bearer <token>`` only.
    ``stream`` reads the answer as SSE (see :func:`_read_stream`).
    """
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if stream:
        body = {**body, "stream": True}
    with requests.post(url, headers=headers, json=body, timeout=timeout_s,
                       stream=stream) as response:
        if response.status_code != 200:
            raise RuntimeError(
                f"API HTTP {response.status_code}: {response.text[:1000]}"
            )
        if not stream:
            try:
                return response.json()
            except ValueError as exc:
                raise RuntimeError(
                    f"API returned non-JSON response: {response.text[:1000]}"
                ) from exc
        return _read_stream(response, model=model)


def extract_response_text(response: dict[str, Any]) -> str:
    """Join the text of the first choice's message content (str or list of parts)."""
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"no choices in GPT response: {json.dumps(response)[:500]}"
        ) from exc
    content = message.get("content")
    if isinstance(content, list):
        text = "".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        )
    else:
        text = "" if content is None else str(content)
    text = text.strip()
    if not text:
        raise RuntimeError(
            f"empty text in GPT response: {json.dumps(response)[:500]}"
        )
    return text


def default_log(message: str) -> None:
    """Default sink for per-task notes (a concurrent sweep buffers these)."""
    print(message, flush=True)


def judge_one(
    task: dict[str, Any],
    model_name: str,
    rules_path: pathlib.Path,
    base_url: str,
    model: str,
    token: str,
    system_override: str | None,
    no_system: bool,
    request_json: bool,
    *,
    fps: float = DEFAULT_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
    quality: int = DEFAULT_JPEG_QUALITY,
    detail: str | None = None,
    image_format: str = "jpeg",
    temperature: float | None = None,
    max_completion_tokens: int | None = None,
    reasoning_effort: str | None = None,
    retries: int = DEFAULT_RETRIES,
    timeout_s: int = 600,
    log: Callable[[str], None] = default_log,
    stream: bool = False,
) -> tuple[dict[str, Any], str]:
    """Run one full judge call: returns (validated payload, model text).

    The media is decoded and the payload rendered once, then sent up to
    ``retries + 1`` times: a transport error, an unusable answer, or a payload
    that fails checklist validation is retried with exponential backoff.  Raises
    on final failure so the caller can log the task as an error and continue.
    ``log`` receives the retry notes; passing a buffer instead of the default
    printer keeps one task's output in a single block when tasks run in parallel.
    """
    spec, level, qas = load_checklist(task["checklist_path"])
    frames, meta = extract_frames(
        task["video_path"], fps=fps, max_frames=max_frames, quality=quality,
        image_format=image_format,
    )
    rendered = build_judge_text(
        spec, level, qas, rules_path, fps=meta["fps"], n_frames=len(frames),
    )
    system_text, user_text = partition_messages(rendered, no_system, system_override)
    body = build_payload(
        model=model, user_text=user_text, frames=frames, system_text=system_text,
        include_frames=True, request_json=request_json, detail=detail,
        image_format=meta["image_format"],
        temperature=temperature, max_completion_tokens=max_completion_tokens,
        reasoning_effort=reasoning_effort,
    )
    attempts = max(0, int(retries)) + 1
    last_error: Exception | None = None
    model_text = ""
    results: dict[str, QAJudgement] = {}
    for attempt in range(attempts):
        if attempt:
            sleep_s = min(RETRY_BACKOFF_S * (2 ** (attempt - 1)), 60.0)
            log(f"  retry {attempt}/{attempts - 1} in {sleep_s:.0f}s "
                f"({type(last_error).__name__}: {last_error})")
            time.sleep(sleep_s)
        try:
            response = call_chat_completions(
                body, base_url=base_url, model=model, token=token,
                timeout_s=timeout_s, stream=stream,
            )
            model_text = extract_response_text(response)
            results = _validate_judge_payload(
                _parse_response(model_text), spec, level, qas,
            )
            break
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            if isinstance(exc, json.JSONDecodeError):
                # Usually the API cut the answer short (reasoning tokens eat the
                # completion budget), so say what to raise if it keeps happening.
                last_error = RuntimeError(
                    f"truncated/unparseable JSON answer ({exc}); retrying -- pass "
                    f"--max-completion-tokens if it repeats"
                )
            else:
                last_error = exc
    else:
        raise RuntimeError(
            f"judge failed after {attempts} attempt(s): "
            f"{type(last_error).__name__}: {last_error}"
        )
    payload = judge_payload(results, spec["task_id"], level)
    payload["judge_model"] = model
    payload["generated_model"] = model_name
    payload["judge_fps"] = meta["fps"]
    payload["judge_frames"] = meta["n_frames"]
    return payload, model_text


# ---------------------------------------------------------------------------
# Model selection (single generator, a list, or a full sweep over all of them)
# ---------------------------------------------------------------------------
ALL_MODEL_ALIASES = ("all", "*", "every", "all-models")


def resolve_model_names(expr: str, videos_root: pathlib.Path = VIDEO_ROOT) -> list[str]:
    """Expand ``--model-name`` into the list of generated models to judge.

    ``all`` / ``*`` selects every model dir under ``data/outputs`` that holds at
    least one task video, sorted by name -- that is the full-sweep mode over all
    generators.  The value may also be a comma-separated list of names
    (``wan2.2,cosmos3-nano``), and the alias may appear inside such a list
    (``all,qwen-video``) -- entries are expanded in order and de-duplicated.
    Explicitly-named missing dirs are kept: the batch loop prints the concrete
    reason per model.
    """
    if not videos_root.is_dir():
        raise SystemExit(f"no generated-video root: {videos_root}")
    tokens = [token.strip() for token in expr.split(",") if token.strip()]
    if not tokens:
        raise SystemExit("--model-name must not be empty")
    every = sorted(
        path.name for path in videos_root.iterdir()
        if path.is_dir() and next(path.glob("*/video.mp4"), None)
    )
    names: list[str] = []
    for token in tokens:
        expanded = every if token.lower() in ALL_MODEL_ALIASES else [token]
        names.extend(name for name in expanded if name not in names)
    if not names:
        raise SystemExit(f"no model dirs with task videos under {videos_root}")
    return names


def select_tasks(
    model_name: str,
    numbers: set[int] | None,
    name_prefixes: list[str] | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Tasks for one generator, capped to ``limit`` (per model, not global)."""
    tasks = collect_tasks(VIDEO_ROOT, model_name, numbers, name_prefixes)
    if limit and limit > 0:
        tasks = tasks[:limit]
    return tasks


def format_duration(seconds: float) -> str:
    """Human-readable elapsed/eta (``45s``, ``12m03s``, ``3h07m``)."""
    seconds = max(0, int(round(seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def run_single_task(
    index: int,
    total: int,
    task: dict[str, Any],
    model_name: str,
    args: argparse.Namespace,
    rules_path: pathlib.Path,
    token: str,
    output_root: pathlib.Path,
    judge_label: str,
) -> tuple[list[str], str]:
    """Judge one task and return its (log block, status); status in generated/skip/error.

    Nothing is printed here: retry notes, ERROR lines and the saved summary are
    buffered so the caller can emit one contiguous block per task even when
    several tasks run concurrently.  Never raises -- a failure becomes an ERROR
    line plus the ``error`` status, so one bad task cannot stop a sweep.
    """
    out_path = _result_path(output_root, judge_label, model_name, task["task_id"])
    lines = [f"[{index}/{total}] judging: {task['task_id']}"]
    if not args.overwrite and out_path.exists():
        return [lines[0].replace("judging:", "skip (exists):")], "skip"

    t0 = time.perf_counter()
    try:
        payload, model_text = judge_one(
            task, model_name, rules_path, args.base_url, args.model, token,
            system_override=args.system, no_system=args.no_system,
            request_json=not args.no_json_request,
            fps=args.fps, max_frames=args.max_frames,
            quality=args.jpeg_quality, detail=args.detail,
            image_format=args.image_format,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens or None,
            reasoning_effort=args.reasoning_effort,
            retries=args.retries, timeout_s=args.timeout,
            log=lines.append,
            stream=args.stream,
        )
    except Exception as exc:  # noqa: BLE001
        lines.append(f"  ERROR ({time.perf_counter() - t0:.0f}s): {exc}")
        return lines, "error"

    elapsed = time.perf_counter() - t0
    if args.raw:
        lines.append("  [raw]\n" + model_text + "\n")
    if args.no_save:
        lines.append("  [no-save]\n"
                     + json.dumps(payload, ensure_ascii=False, indent=2))
        return lines, "generated"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    tally = Counter(r["verdict"] for r in payload["qa_results"])
    lines.append(f"  -> ok in {elapsed:.0f}s  ({payload['judge_frames']} frames "
                 f"@ {payload['judge_fps']:g} fps)  (yes={tally['yes']} "
                 f"no={tally['no']} unjudgeable={tally['unjudgeable']})")
    return lines, "generated"


def judge_model_tasks(
    model_name: str,
    tasks: list[dict[str, Any]],
    args: argparse.Namespace,
    rules_path: pathlib.Path,
    token: str,
    output_root: pathlib.Path,
    judge_label: str,
) -> tuple[int, int, int]:
    """Judge one generator's task list.  Returns (generated, skipped, errors).

    Tasks whose result JSON already exist are skipped unless ``--overwrite``, so a
    large sweep is resumable: rerun the same command and it continues where it
    stopped.  ``--workers N`` runs N tasks at once (API calls dominate the wall
    clock, so this is nearly linear speed-up); blocks are still printed one per
    task from this thread, in completion order, each carrying its own
    ``[index/total]`` so an out-of-order line is never ambiguous.  A failing task
    prints ERROR and does not stop the sweep.
    """
    total = len(tasks)
    generated = skipped = errors = 0
    started_at = time.perf_counter()
    workers = max(1, int(args.workers or 1))

    def emit(lines: list[str], tail: str = "") -> None:
        # flush=True so progress lines appear live even when stdout is piped/a file.
        for line in lines:
            print(line, flush=True)
        if tail:
            print(f"  [{tail}]", flush=True)

    if workers == 1:
        for index, task in enumerate(tasks, start=1):
            lines, status = run_single_task(
                index, total, task, model_name, args, rules_path, token,
                output_root, judge_label,
            )
            generated += status == "generated"
            skipped += status == "skip"
            errors += status == "error"
            emit(lines)
        return generated, skipped, errors

    # Concurrent: the main thread only counts and prints, workers only call the API.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(run_single_task, index, total, task, model_name, args,
                        rules_path, token, output_root, judge_label)
            for index, task in enumerate(tasks, start=1)
        ]
        done = 0
        for future in as_completed(futures):
            lines, status = future.result()
            generated += status == "generated"
            skipped += status == "skip"
            errors += status == "error"
            done += 1
            # No tail for instant skips: it would only repeat a meaningless eta.
            tail = ""
            if status != "skip" and done < total:
                rate = (time.perf_counter() - started_at) / max(1, done - skipped)
                remaining = total - done
                tail = (f"done {done}/{total}, generated={generated}, "
                        f"errors={errors}, eta {format_duration(rate * remaining)}")
            emit(lines, tail)
    return generated, skipped, errors

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Judge generated videos (single task or batch) with a GPT "
                    "multimodal judge API -- frames sampled at --fps and sent as "
                    "base64 image_url parts -- against frozen QA checklists"
    )
    p.add_argument("--model-name", required=True,
                   help="generated-model name; videos read from "
                        "data/outputs/<model-name>/<task-id>/video.mp4.  'all' "
                        "(or '*') sweeps every model dir under data/outputs; a "
                        "comma-separated list ('wan2.2,cosmos3-nano') judges a "
                        "subset")
    p.add_argument("--task-id", default=None,
                   help="select tasks by leading task number / range / prefix, "
                        "comma-separated: '001' one, '001-020' a range, "
                        "'001,003,005' discrete, or a full/partial dir name like "
                        "'001_L1_001_agronomy' (default: every task under the "
                        "model dir)")
    p.add_argument("--rules", default=str(DEFAULT_RULES),
                   help="frozen judge rules template (default: data/rules/vlm_judge.txt)")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL,
                   help="OpenAI-compatible base url; required, or set "
                        "JUDGE_API_BASE_URL")
    p.add_argument("--model", default="",
                   help="judge model id served by your endpoint; required for a "
                        "real call -- no default is assumed")
    p.add_argument("--token", default=None,
                   help="bearer token sent as Authorization: Bearer <token> "
                        "(env GPT_API_TOKEN / OPENAI_API_KEY)")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS,
                   help="keyframes sampled per second of video (default: "
                        "%(default)s; the answers bucket name follows this value)")
    p.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
                   help="cap the frame payload; long clips get a lower effective "
                        "fps instead of being truncated (0 = no cap, default: "
                        f"{DEFAULT_MAX_FRAMES})")
    p.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY,
                   help="ffmpeg -q:v for the extracted frames, 1 (best) to 31 "
                        f"(worst); ignored by --image-format png "
                        f"(default: {DEFAULT_JPEG_QUALITY})")
    p.add_argument("--image-format", default="jpeg", choices=["jpeg", "png"],
                   help="frame encoding sent as base64 image_url parts (default: "
                        "jpeg; png is lossless but ~9-11x the payload on this "
                        "data: a 12s/48-frame clip grows from ~2 MB (2.6 MB "
                        "base64) to ~22 MB (29 MB base64), which slow endpoints "
                        "reject, while the billed image tokens stay the same "
                        "because the API tiles by resolution, not file size)")
    p.add_argument("--detail", default=None,
                   choices=["auto", "low", "high", "original"],
                   help="image_url.detail for every frame (default: the "
                        "provider's own auto behaviour)")
    p.add_argument("--temperature", type=float, default=None,
                   help="send temperature (default: omit it; gpt-5.x reasoning "
                        "models ignore or reject values other than 1)")
    p.add_argument("--max-completion-tokens", type=int, default=0,
                   help="cap completion tokens including reasoning (0 = omit the "
                        "field, let the server decide; raise it when the log "
                        "reports truncated/unparseable JSON answers, which is what "
                        "a too-small completion budget looks like here)")
    p.add_argument("--reasoning-effort", default=None,
                   choices=["minimal", "low", "medium", "high"],
                   help="reasoning_effort for gpt-5.x judges (default: server "
                        "default)")
    p.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                   help="extra attempts per task after a transport error or an "
                        "invalid judge answer (0 = off, default: "
                        f"{DEFAULT_RETRIES})")
    p.add_argument("--timeout", type=int, default=600,
                   help="per-request timeout in seconds (default: 600)")
    p.add_argument("--stream", action="store_true",
                   help="read the answer as an SSE stream: bytes keep flowing "
                        "while the model reasons, so a long call cannot look "
                        "like a stalled connection")
    p.add_argument("--workers", type=int, default=4,
                   help="judge this many tasks of the same model at once "
                        "(1 = sequential, the judge_videos_gemini.py parity "
                        "behaviour; 4-8 is the usual sweet spot for a judge "
                        "sweep -- raise it only as far as the endpoint's rate "
                        "limit allows)")
    p.add_argument("--system", default=None,
                   help="override system message (default: vlm_judge.txt "
                        "paragraph 1, the role framing)")
    p.add_argument("--no-system", action="store_true",
                   help="omit the system message; send the whole rendered judge "
                        "text as one user turn (Qwen-judge parity)")
    p.add_argument("--output-dir", default=None,
                   help="answers dir; each result saved to "
                        "<dir>/<judge-label>/<model-name>/<task-id>.json "
                        "(default: data/results/fps<--fps>)")
    p.add_argument("--judge-label", default=None,
                   help="judge label used as the results subfolder "
                        "(default: the --model id)")
    p.add_argument("--overwrite", action="store_true",
                   help="re-judge tasks whose result JSON already exists "
                        "(default: skip them)")
    p.add_argument("--limit", type=int, default=0,
                   help="judge at most this many tasks per model (0 = no limit; "
                        "with --model-name all the cap applies to each "
                        "generator separately)")
    p.add_argument("--prompt-only", action="store_true",
                   help="print the rendered judge prompt of the first selected "
                        "task and exit (no media read, no API call)")
    p.add_argument("--dry-run", action="store_true",
                   help="build the payload of the first selected task and show "
                        "its structure, no network call")
    p.add_argument("--raw", action="store_true",
                   help="also print each model text before parsing")
    p.add_argument("--no-save", action="store_true",
                   help="do not write result files (payloads still printed)")
    p.add_argument("--no-json-request", action="store_true",
                   help="do not set response_format=json_object "
                        "(endpoints that reject it)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rules_path = pathlib.Path(args.rules).expanduser().resolve()
    if not rules_path.is_file():
        raise SystemExit(f"rules template not found: {rules_path}")

    numbers, name_prefixes = (None, None)
    if args.task_id:
        try:
            numbers, name_prefixes = parse_task_selection(args.task_id)
        except ValueError as exc:
            raise SystemExit(f"bad --task-id {args.task_id!r}: {exc}") from exc
    model_names = resolve_model_names(args.model_name)
    judge_label = args.judge_label or args.model

    if args.prompt_only or args.dry_run:
        # Media inspection is per-task: show the first task of the first model.
        shown = model_names[0]
        if len(model_names) > 1:
            print(f"note: --{'prompt-only' if args.prompt_only else 'dry-run'} inspects "
                  f"the first model only ({shown}); pass --model-name <name> to look at "
                  f"the others", file=sys.stderr, flush=True)
        try:
            tasks = select_tasks(shown, numbers, name_prefixes, args.limit)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
        first = tasks[0]
        spec, level, qas = load_checklist(first["checklist_path"])
        if args.prompt_only:
            print(build_judge_text(spec, level, qas, rules_path, fps=args.fps))
            return 0
        frames, meta = extract_frames(
            first["video_path"], fps=args.fps, max_frames=args.max_frames,
            quality=args.jpeg_quality, image_format=args.image_format,
        )
        rendered = build_judge_text(
            spec, level, qas, rules_path, fps=meta["fps"], n_frames=len(frames),
        )
        system_text, user_text = partition_messages(rendered, args.no_system, args.system)
        body = build_payload(
            model=args.model, user_text=user_text, frames=frames,
            system_text=system_text, include_frames=False,
            request_json=not args.no_json_request, detail=args.detail,
            image_format=args.image_format,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens or None,
            reasoning_effort=args.reasoning_effort,
        )
        print("=== system message ===")
        print(system_text)
        print("\n=== payload structure (frame bytes omitted) ===")
        print(json.dumps(body, ensure_ascii=False, indent=2))
        print(f"\nmodel: {shown}  task: {first['task_id']}")
        print(f"video: {first['video_path']}  "
              f"({first['video_path'].stat().st_size:,} bytes, {meta['duration']:.2f}s)")
        print(f"frames: {meta['n_frames']} {meta['image_format']} @ {meta['fps']:g} fps "
              f"({meta['image_bytes']:,} bytes, "
              f"{sum(len(f) for f in frames):,} base64 chars)")
        print(f"prompt chars: {len(user_text):,}  |  qa items: {len(qas)}  "
              f"|  level: {level}")
        return 0

    if not args.base_url:
        raise SystemExit("no API endpoint: pass --base-url or set JUDGE_API_BASE_URL")
    if not args.model:
        raise SystemExit(
            "no judge model: pass --model with the model id your endpoint serves"
        )
    token = (args.token or os.environ.get("GPT_API_TOKEN")
             or os.environ.get("OPENAI_API_KEY"))
    if not token:
        raise SystemExit(
            "no API credentials: set GPT_API_TOKEN (or pass --token)"
        )

    output_root = pathlib.Path(
        args.output_dir or _answers_dir(args.fps)
    ).expanduser().resolve()
    totals: Counter = Counter()
    started_at = time.perf_counter()
    sweep = len(model_names) > 1
    blank = "\n" if sweep else ""
    # flush=True so progress lines appear live even when stdout is piped/a file.
    if sweep:
        print(f"models({len(model_names)})={','.join(model_names)}  "
              f"judge={judge_label}  fps={args.fps:g}  workers={args.workers}  "
              f"-> {output_root}", flush=True)
    for model_index, model_name in enumerate(model_names, start=1):
        try:
            tasks = select_tasks(model_name, numbers, name_prefixes, args.limit)
        except FileNotFoundError as exc:
            if (VIDEO_ROOT / model_name).is_dir():
                print(f"{blank}[{model_index}/{len(model_names)}] {model_name}: nothing "
                      f"to judge for this --task-id selection", flush=True)
                totals["models_empty"] += 1
            else:
                print(f"{blank}[{model_index}/{len(model_names)}] {model_name}: FAILED "
                      f"({exc})", flush=True)
                totals["models_failed"] += 1
            continue
        print(f"{blank}[{model_index}/{len(model_names)}] model={model_name}  "
              f"tasks selected={len(tasks)}  judge={judge_label}  "
              f"fps={args.fps:g}  workers={args.workers}  -> {output_root}", flush=True)
        generated, skipped, errors = judge_model_tasks(
            model_name, tasks, args, rules_path, token, output_root, judge_label,
        )
        totals.update(generated=generated, skipped=skipped, errors=errors,
                      selected=len(tasks), models=1)

    total_elapsed = time.perf_counter() - started_at
    avg = (total_elapsed / totals["generated"]) if totals["generated"] else 0.0
    print(f"\nDone. models={totals['models']}  selected={totals['selected']}  "
          f"generated={totals['generated']}  "
          f"skipped(existing)={totals['skipped']}  errors={totals['errors']}  "
          f"elapsed={total_elapsed:.0f}s"
          f"{f'  avg={avg:.1f}s/task' if totals['generated'] else ''}",
          flush=True)
    if totals["models_empty"]:
        print(f"note: {totals['models_empty']} of {len(model_names)} model(s) had no task "
              f"matching the --task-id selection", flush=True)
    return 0 if totals["errors"] == 0 and totals["models_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
