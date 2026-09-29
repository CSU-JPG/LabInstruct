#!/usr/bin/env python3
"""Judge generated videos with a Gemini multimodal API against frozen QA
checklists — single task or batch.

Videos are read from ``data/outputs/<model-name>/<task-id>/video.mp4`` and the
matching checklist from ``data/checklists/<task-id>_qa.json`` (1:1 by task id).
Pass ``--model-name wan2.2`` to select a generator; add ``--task-id`` to select
tasks by their globally-unique leading number (``001`` one, ``001-020`` an
inclusive range, ``001,003,005`` discrete), or by a full/partial dir name such
as ``001_L1_001_agronomy``.  Without ``--task-id``, every task under the model
dir is judged in batch.

The judge prompt is rendered from data/rules/vlm_judge.txt exactly as in
bench.eval.vlm_judge.build_judge_prompt: paragraph 1 becomes the Gemini
system_instruction, the rest (embedded checklist + rules + output format) is the
user prompt.  The generated video is attached as an inline base64 video part
(``inline_data.data``).  The model's OUTPUT-FORMAT JSON answer is validated
against the frozen checklist and saved to

    <output-dir>/<judge-label>/<model-name>/<task-id>.json

with ``<output-dir>`` defaulting to data/results/fps<FPS> and ``<judge-label>``
to the API model id you pass with --model (e.g.
``data/results/fps4/<model>/wan2.2/001_L1_001_agronomy.json``).

Credentials.  Ignored in --prompt-only/--dry-run; a real call authenticates with
exactly one bearer token, read from the environment (never stored in this file):
    GEMINI_API_TOKEN  -> sent as  Authorization: Bearer <token>
You may also pass --token explicitly.

Endpoint.  The API host is not hardcoded: pass --host or set JUDGE_API_HOST.
Any host serving the Gemini generateContent schema works.

Examples:
    # Inspect the judge prompt for one task (no API call, no media read)
    python scripts/judge_videos_gemini.py --model-name wan2.2 --task-id 001 --prompt-only

    # Show the payload structure for one task (no API call)
    python scripts/judge_videos_gemini.py --model-name wan2.2 --task-id 001 --dry-run

    # Judge a single task, a range, or several discrete numbers of wan2.2
    python scripts/judge_videos_gemini.py --model-name wan2.2 --task-id 001
    python scripts/judge_videos_gemini.py --model-name wan2.2 --task-id 001-020
    python scripts/judge_videos_gemini.py --model-name wan2.2 --task-id 001,003,005

    # Judge every task under data/outputs/wan2.2 in batch
    python scripts/judge_videos_gemini.py --model-name wan2.2

    # Batch but cap to the first 3 (smoke test); re-judge instead of skipping
    python scripts/judge_videos_gemini.py --model-name wan2.2 --limit 3
    python scripts/judge_videos_gemini.py --model-name wan2.2 --overwrite
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import os
import pathlib
import re
import sys
import time
from collections import Counter
from typing import Any

# Make `bench` importable when launched as a plain script.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from bench.eval.qa import _parse_response, load_qas  # noqa: E402
from bench.eval.vlm_judge import (  # noqa: E402
    QAItem,
    _validate_judge_payload,
    build_judge_prompt,
    judge_payload,
)

# ---------------------------------------------------------------------------
# Path defaults (the data/ layout shipped with the benchmark)
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
VIDEO_ROOT = DATA_DIR / "outputs"
CHECKLIST_DIR = DATA_DIR / "checklists"
DEFAULT_RULES = DATA_DIR / "rules" / "vlm_judge.txt"
DEFAULT_FPS = 4

# No API host is hardcoded: pass --host or export JUDGE_API_HOST.  Any endpoint
# serving the Gemini generateContent schema works.
DEFAULT_HOST = os.environ.get("JUDGE_API_HOST", "")

# No credential is stored in this file.  Export GEMINI_API_TOKEN, or pass --token.
# The answers bucket name follows --fps, so it is resolved in main(), not here.
def _answers_dir(fps: float) -> pathlib.Path:
    """Answers bucket for one sampling rate: data/results/fps<--fps>/, which is
    how the records shipped with this repository are laid out."""
    return DATA_DIR / "results" / f"fps{fps:g}"

# Media marker for the {{GENERATED_VIDEO}} placeholder.  The video bytes travel
# as a separate inline_data part; the text only describes where they live.
VIDEO_TEXT = "Provided as the video part of this message (the generated video)."

_MIME_BY_SUFFIX = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
}


# ---------------------------------------------------------------------------
# Task discovery
# ---------------------------------------------------------------------------
def _lead_number(dir_name: str) -> int | None:
    """The globally-unique leading task number, e.g. 1 from ``001_L1_...``."""
    head = dir_name.split("_", 1)[0]
    return int(head) if head.isdigit() else None


def parse_task_selection(expr: str) -> tuple[set[int], list[str]]:
    """Expand a --task-id expression into (lead numbers, dir-name prefixes).

    Comma separates items; each item is either
      * a discrete number, e.g. ``001``   (matched against the leading task number)
      * an inclusive range, e.g. ``001-020``
      * a directory-name prefix / full task id, e.g. ``001_L1_001_agronomy``
    Mixed forms work, e.g. ``001-003,005,001_L1_001_agronomy``.
    Raises ValueError on malformed items.
    """
    numbers: set[int] = set()
    names: list[str] = []
    for raw in expr.split(","):
        token = raw.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, _, hi_s = token.partition("-")
            if not (lo_s.strip().isdigit() and hi_s.strip().isdigit()):
                raise ValueError(f"bad range {token!r} (want digits-digits)")
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                raise ValueError(f"bad range {token!r} (start > end)")
            numbers.update(range(lo, hi + 1))
        elif token.isdigit():
            numbers.add(int(token))
        else:
            names.append(token)
    return numbers, names


def _selection_matches(name: str, numbers: set[int], name_prefixes: list[str]) -> bool:
    lead = _lead_number(name)
    if lead is not None and lead in numbers:
        return True
    return any(name == pre or name.startswith(pre) for pre in name_prefixes)


def collect_tasks(
    videos_root: pathlib.Path,
    model_name: str,
    numbers: set[int] | None = None,
    name_prefixes: list[str] | None = None,
    checklist_dir: pathlib.Path = CHECKLIST_DIR,
) -> list[dict[str, Any]]:
    """Collect judgeable tasks under ``<videos_root>/<model_name>``.

    A task is judgeable when its directory holds ``video.mp4`` and a matching
    ``<task_id>_qa.json`` exists in ``checklist_dir``.  Selection: ``numbers``
    filters by leading task number, ``name_prefixes`` by dir-name prefix/full id;
    both ``None`` selects every task under the model dir.  Raises
    FileNotFoundError when the model dir is missing or nothing matched.
    """
    model_dir = videos_root / model_name
    if not model_dir.is_dir():
        raise FileNotFoundError(f"no generated videos for model {model_name}: {model_dir}")
    numbers = numbers or set()
    name_prefixes = name_prefixes or []
    selected = numbers or name_prefixes  # empty selection == no filter
    dirs = sorted(
        p for p in model_dir.iterdir()
        if p.is_dir() and (not selected or _selection_matches(p.name, numbers, name_prefixes))
    )
    tasks: list[dict[str, Any]] = []
    missing = 0
    for d in dirs:
        video = d / "video.mp4"
        checklist = checklist_dir / f"{d.name}_qa.json"
        if video.is_file() and checklist.is_file():
            tasks.append({"task_id": d.name, "video_path": video,
                          "checklist_path": checklist})
        else:
            missing += 1
    if missing:
        print(f"note: skipped {missing} task dir(s) without video.mp4 or checklist")
    if not tasks:
        raise FileNotFoundError(
            f"no judgeable tasks for model {model_name} under {model_dir}"
            + (f" matching {numbers or name_prefixes!r}" if selected else "")
        )
    return tasks


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_checklist(path: pathlib.Path) -> tuple[dict[str, Any], str, list[QAItem]]:
    """Read a *_qa.json checklist -> (spec metadata, task level, QA items)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    task_id = str(payload.get("task_id", "")).strip()
    if not task_id:
        raise ValueError(f"checklist {path} must contain a non-empty task_id")
    level = str(payload.get("task_level", "")).strip().upper() or _level_from_id(task_id)
    qas = load_qas(path)
    if not qas:
        raise ValueError(f"checklist {path} contains no QA items")
    spec = {"task_id": task_id}
    return spec, level, qas


def _level_from_id(task_id: str) -> str:
    match = re.search(r"(?:^|_)(L[12])(?:_|$)", task_id.upper())
    if not match:
        raise ValueError(f"cannot parse task level from task_id {task_id!r}")
    return match.group(1)


def mime_type_for(video_path: pathlib.Path) -> str:
    return _MIME_BY_SUFFIX.get(video_path.suffix.lower(), "video/mp4")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
def build_judge_text(
    spec: dict[str, Any],
    level: str,
    qas: list[QAItem],
    rules_path: pathlib.Path = DEFAULT_RULES,
) -> str:
    """Render the frozen vlm_judge.txt rule + embedded checklist (no media)."""
    return build_judge_prompt(
        spec, qas, level, video_text=VIDEO_TEXT, template_path=rules_path
    )


def split_system_user(rendered: str) -> tuple[str, str]:
    """Split a rendered judge prompt into (system, user) at its first blank line.

    vlm_judge.txt opens with one role paragraph ("You are evaluating a generated
    laboratory instructional video ... your only role is to judge whether each QA
    requirement is visibly satisfied ...").  That paragraph reads naturally as a
    Gemini system_instruction; everything after it (INPUTS with the embedded
    checklist, per-dimension rules, OUTPUT FORMAT, strict requirements) becomes
    the user prompt.  Paragraph 1 contains no template placeholders, so the split
    is safe after rendering.  Falls back to ("", rendered) when there is no clean
    first paragraph.
    """
    parts = rendered.split("\n\n", 1)
    if len(parts) == 2 and parts[0].strip():
        return parts[0].strip(), parts[1].lstrip("\n")
    return "", rendered


def partition_messages(
    rendered: str, no_system: bool, system_override: str | None
) -> tuple[str | None, str]:
    """Turn a rendered judge prompt into (system_text, user_text).

    Paragraph 1 -> system_instruction; the rest -> user prompt.  ``no_system``
    re-sends the whole rendered text as a single user turn (Qwen-judge parity);
    ``system_override`` replaces the system text only.
    """
    default_system, default_user = split_system_user(rendered)
    if no_system:
        return None, rendered
    system = system_override or default_system or None
    user = default_user or rendered
    return system, user


# ---------------------------------------------------------------------------
# Video -> base64 (bytes never inlined in source; referenced by variable)
# ---------------------------------------------------------------------------
def encode_video_base64(video_path: pathlib.Path) -> str:
    """Read a video file and return its base64 payload for inline_data.data."""
    video_bytes = video_path.read_bytes()
    return base64.b64encode(video_bytes).decode("ascii")


# ---------------------------------------------------------------------------
# Gemini generateContent payload
# ---------------------------------------------------------------------------
def build_payload(
    *,
    user_text: str,
    video_path: pathlib.Path,
    mime_type: str,
    system_text: str | None = None,
    include_video: bool = True,
    request_json: bool = True,
    fps: float = DEFAULT_FPS,
) -> dict[str, Any]:
    """Build the generateContent body (a dict; json.dumps happens at send time).

    With ``include_video=False`` the inline_data part is replaced by a text
    placeholder so --dry-run can show the structure without a megabytes-long
    base64 string.  The media is placed first and the instruction text after it.

    ``request_json`` adds generationConfig.responseMimeType=application/json so
    the model returns the OUTPUT-FORMAT object as raw JSON instead of wrapping it
    in Markdown/commentary (per data/rules/vlm_judge.txt).  Some endpoints reject
    the field; disable it with --no-json-request and the text rules alone still
    ask for JSON.
    """
    if include_video:
        data_b64 = encode_video_base64(video_path)
        media_part = {
            "inline_data": {
                "mime_type": mime_type, 
                "data": data_b64, 
                "processing": {
                    "type": "static",
                    "fps": fps
                }
                }
            }
    else:
        media_part = {"text": f"[video inline_data omitted; {video_path.stat().st_size} bytes, mime {mime_type}]"}

    contents = [{"role": "user", "parts": [media_part, {"text": user_text}]}]
    body: dict[str, Any] = {"contents": contents}
    if system_text:
        body["system_instruction"] = {"parts": [{"text": system_text}]}
    config: dict[str, Any] = {"temperature": 0.0}
    if request_json:
        config["responseMimeType"] = "application/json"
    body["generationConfig"] = config
    return body


def call_generate_content(
    body: dict[str, Any],
    *,
    host: str,
    model: str,
    token: str | None,
    timeout_s: int = 600,
) -> dict[str, Any]:
    """POST the payload to <host>/v1beta/models/<model>:generateContent.

    Authenticates with ``Authorization: Bearer <token>`` only (no ?key= query).
    """
    path = f"/v1beta/models/{model}:generateContent"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    conn = http.client.HTTPSConnection(host, timeout=timeout_s)
    try:
        conn.request("POST", path, body=json.dumps(body), headers=headers)
        res = conn.getresponse()
        raw = res.read().decode("utf-8", errors="replace")
    finally:
        conn.close()
    if res.status != 200:
        raise RuntimeError(f"API HTTP {res.status} {res.reason}: {raw[:1000]}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"API returned non-JSON response: {raw[:1000]}") from exc


def extract_response_text(response: dict[str, Any]) -> str:
    """Join the text of the first candidate's content parts."""
    try:
        parts = response["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"no candidates in Gemini response: {json.dumps(response)[:500]}") from exc
    text = "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict)).strip()
    if not text:
        raise RuntimeError(f"empty text in Gemini response: {json.dumps(response)[:500]}")
    return text


def judge_one(
    task: dict[str, Any],
    model_name: str,
    rules_path: pathlib.Path,
    host: str,
    model: str,
    token: str,
    system_override: str | None,
    no_system: bool,
    request_json: bool,
    fps: float = DEFAULT_FPS,
) -> tuple[dict[str, Any], str]:
    """Run one full judge call: returns (validated payload, model text).

    Raises on any failure so the caller can catch per-task errors and continue.
    """
    spec, level, qas = load_checklist(task["checklist_path"])
    rendered = build_judge_text(spec, level, qas, rules_path)
    system_text, user_text = partition_messages(rendered, no_system, system_override)
    body = build_payload(
        user_text=user_text, video_path=task["video_path"],
        mime_type=mime_type_for(task["video_path"]),
        system_text=system_text, include_video=True, request_json=request_json,
        fps=fps,
    )
    response = call_generate_content(
        body, host=host, model=model, token=token
    )
    model_text = extract_response_text(response)
    parsed = _parse_response(model_text)
    results = _validate_judge_payload(parsed, spec, level, qas)
    payload = judge_payload(results, spec["task_id"], level)
    payload["judge_model"] = model
    payload["generated_model"] = model_name
    return payload, model_text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Judge generated videos (single task or batch) with a Gemini "
                    "multimodal judge API against frozen QA checklists"
    )
    p.add_argument("--model-name", required=True,
                   help="generated-model name; videos read from "
                        "data/outputs/<model-name>/<task-id>/video.mp4")
    p.add_argument("--task-id", default=None,
                   help="select tasks by leading task number / range / prefix, "
                        "comma-separated: '001' one, '001-020' a range, "
                        "'001,003,005' discrete, or a full/partial dir name like "
                        "'001_L1_001_agronomy' (default: every task under the "
                        "model dir)")
    p.add_argument("--rules", default=str(DEFAULT_RULES),
                   help="frozen judge rules template (default: data/rules/vlm_judge.txt)")
    p.add_argument("--host", default=DEFAULT_HOST,
                   help="API host serving the Gemini generateContent schema; "
                        "required, or set JUDGE_API_HOST")
    p.add_argument("--model", default="",
                   help="judge model id served by your endpoint; required for a "
                        "real call -- no default is assumed")
    p.add_argument("--token", default=None,
                   help="bearer token sent as Authorization: Bearer <token> "
                        "(env GEMINI_API_TOKEN)")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS,
                   help="video sampling rate handed to the API as "
                        "processing.fps (default: %(default)s); the answers "
                        "bucket name follows this value")
    p.add_argument("--system", default=None,
                   help="override system_instruction (default: vlm_judge.txt "
                        "paragraph 1, the role framing)")
    p.add_argument("--no-system", action="store_true",
                   help="omit system_instruction; send the whole rendered judge "
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
                   help="judge at most this many tasks (0 = no limit)")
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
                   help="do not set responseMimeType=application/json "
                        "(endpoints that reject it)")
    return p


def _result_path(output_dir: pathlib.Path, judge_label: str, model_name: str,
                 task_id: str) -> pathlib.Path:
    return output_dir / judge_label / model_name / f"{task_id}.json"


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
    try:
        tasks = collect_tasks(VIDEO_ROOT, args.model_name, numbers, name_prefixes)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    if args.limit and args.limit > 0:
        tasks = tasks[:args.limit]
    judge_label = args.judge_label or args.model

    first = tasks[0]
    if args.prompt_only:
        spec, level, qas = load_checklist(first["checklist_path"])
        print(build_judge_text(spec, level, qas, rules_path))
        return 0
    if args.dry_run:
        spec, level, qas = load_checklist(first["checklist_path"])
        rendered = build_judge_text(spec, level, qas, rules_path)
        system_text, user_text = partition_messages(rendered, args.no_system, args.system)
        body = build_payload(
            user_text=user_text, video_path=first["video_path"],
            mime_type=mime_type_for(first["video_path"]),
            system_text=system_text, include_video=False,
        )
        print("=== system_instruction ===")
        print(system_text)
        print("\n=== payload structure (video bytes omitted) ===")
        print(json.dumps(body, ensure_ascii=False, indent=2))
        print(f"\nvideo: {first['video_path']}  ({first['video_path'].stat().st_size:,} bytes, "
              f"{mime_type_for(first['video_path'])})")
        print(f"prompt chars: {len(user_text):,}  |  qa items: {len(qas)}  "
              f"|  level: {level}")
        return 0

    if not args.host:
        raise SystemExit(
            "no API host: pass --host or set JUDGE_API_HOST"
        )
    if not args.model:
        raise SystemExit(
            "no judge model: pass --model with the model id your endpoint serves"
        )
    token = args.token or os.environ.get("GEMINI_API_TOKEN")
    if not token:
        raise SystemExit(
            "no API credentials: set GEMINI_API_TOKEN (or pass --token)"
        )

    output_root = pathlib.Path(
        args.output_dir or _answers_dir(args.fps)
    ).expanduser().resolve()
    generated = skipped = errors = 0
    total = len(tasks)
    started_at = time.perf_counter()
    # flush=True so progress lines appear live even when stdout is piped/a file.
    print(f"model={args.model_name}  tasks selected={total}  "
          f"judge={judge_label}  -> {output_root}", flush=True)
    for index, task in enumerate(tasks, start=1):
        out_path = _result_path(output_root, judge_label, args.model_name, task["task_id"])
        if not args.overwrite and out_path.exists():
            print(f"[{index}/{total}] skip (exists): {task['task_id']}", flush=True)
            skipped += 1
            continue
        t0 = time.perf_counter()
        print(f"[{index}/{total}] judging: {task['task_id']}", flush=True)
        try:
            payload, model_text = judge_one(
                task, args.model_name, rules_path, args.host, args.model, token,
                system_override=args.system, no_system=args.no_system,
                request_json=not args.no_json_request, fps=args.fps,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR ({time.perf_counter() - t0:.0f}s): {exc}", flush=True)
            errors += 1
            continue
        elapsed = time.perf_counter() - t0
        if args.raw:
            print("  [raw]\n" + model_text + "\n", flush=True)
        if not args.no_save:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            tally = Counter(r["verdict"] for r in payload["qa_results"])
            print(f"  -> ok in {elapsed:.0f}s  (yes={tally['yes']} "
                  f"no={tally['no']} unjudgeable={tally['unjudgeable']})",
                  flush=True)
        else:
            print("  [no-save]\n" + json.dumps(payload, ensure_ascii=False, indent=2),
                  flush=True)
        generated += 1

    total_elapsed = time.perf_counter() - started_at
    avg = (total_elapsed / generated) if generated else 0.0
    print(f"\nDone. selected={total}  generated={generated}  "
          f"skipped(existing)={skipped}  errors={errors}  "
          f"elapsed={total_elapsed:.0f}s"
          f"{f'  avg={avg:.1f}s/task' if generated else ''}",
          flush=True)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
