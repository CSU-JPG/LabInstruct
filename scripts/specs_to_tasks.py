#!/usr/bin/env python
"""Convert LabInstruct spec annotations into the harness task format.

Input : directory of spec JSON files. The canonical spec schema is the one in
        data/specs/ (see data/specs/*_spec.json): a LabInstruct
        annotation with ``task_id``, ``split`` ("L1"/"L2"), ``initial_image``
        (relative path to the first frame I0), ``prompt_for_gen`` (the
        generation instruction), and the structured ``scene`` +
        ``action_sequence`` + ``difficulty_features``.
Output: a tasks.jsonl consumable by ``python -m bench.cli gen``. Each task is
        a thin, source-faithful projection: only the fields the harness
        actually consumes are emitted, and ``spec`` mirrors the source spec's
        structure verbatim — state transitions stay inside each action step
        (they are not flattened out into a duplicate top-level list).

Generation settings (frames / fps / resolution) are NOT part of the task —
they are per-model, read by each adapter from models.yaml (paper Table 6).
The task only carries the ``level`` (1/2) that selects L1/L2 frame counts.

Usage:
    python scripts/specs_to_tasks.py   # paths hardcoded at module top
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bench import PROJECT_ROOT  # noqa: E402

# Defaults; override with --specs / --out.
SPECS_DIR = str(PROJECT_ROOT / "data" / "specs")
TASKS_OUT = str(PROJECT_ROOT / "data" / "tasks.jsonl")
IMAGE_SUBDIR = "first_frames"

_LEVEL_BY_SPLIT = {"L1": 1, "L2": 2}


def _split_level(split: str | None) -> tuple[str, int]:
    """Normalize a spec ``split`` to (key, level); unknown split is an error.

    The previous code silently fell back to L2 level but L1 frame counts for
    unrecognized splits — that inconsistency made the task self-contradictory.
    """
    key = (split or "L1").strip().upper()
    if key not in _LEVEL_BY_SPLIT:
        raise ValueError(f"unsupported split {split!r} (expected L1 or L2)")
    return key, _LEVEL_BY_SPLIT[key]


def spec_to_task(
    spec: dict,
    *,
    image_subdir: str,
) -> dict:
    """Convert one spec annotation dict into a harness task dict."""
    task_id = spec.get("task_id")
    if not task_id:
        raise ValueError(f"spec missing task_id: {json.dumps(spec, ensure_ascii=False)[:200]}")

    _, level = _split_level(spec.get("split", "L1"))

    prompt_for_gen = spec.get("prompt_for_gen", "").strip()
    if not prompt_for_gen:
        raise ValueError(
            f"spec {task_id!r} has no prompt_for_gen "
            "(the generation instruction is required)"
        )

    # I0 (first frame): prefer the explicit spec field; fall back to the
    # conventional spec-dir-relative path ../first_frames/<task_id>.jpg.
    initial_image = spec.get("initial_image") or spec.get("initial_frame")
    if not initial_image:
        initial_image = f"../{image_subdir}/{task_id}.jpg"

    return {
        "task_id": task_id,
        "level": level,
        "prompt_for_gen": prompt_for_gen,
        "initial_frame": initial_image,
        # Mirror the source spec verbatim. State transitions live inside each
        # action_sequence step and are NOT duplicated as a top-level list.
        "spec": {
            "scene": spec.get("scene", {}),
            "action_sequence": spec.get("action_sequence", []),
            "difficulty_features": spec.get("difficulty_features", {}),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert spec annotations into the harness task format."
    )
    parser.add_argument("--specs", type=pathlib.Path, default=pathlib.Path(SPECS_DIR),
                        help=f"directory of spec JSON files (default: {SPECS_DIR})")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path(TASKS_OUT),
                        help=f"output tasks.jsonl (default: {TASKS_OUT})")
    parser.add_argument("--image-subdir", default=IMAGE_SUBDIR,
                        help="directory holding the first frames, relative to the "
                             f"spec file (default: {IMAGE_SUBDIR})")
    args = parser.parse_args(argv)

    specs_dir = args.specs.expanduser().resolve()
    spec_files = sorted(specs_dir.glob("*.json"))
    if not spec_files:
        print(f"no spec json files under {specs_dir}", file=sys.stderr)
        return 1

    tasks = []
    for p in spec_files:
        spec = json.loads(p.read_text())
        task = spec_to_task(spec, image_subdir=args.image_subdir)
        # Spec initial_image paths are relative to the spec file's directory;
        # re-express them relative to PROJECT_ROOT so the harness (which
        # resolves initial_frame against PROJECT_ROOT) finds the images.
        frame = task["initial_frame"]
        resolved = (p.parent / frame).resolve()
        try:
            task["initial_frame"] = str(resolved.relative_to(PROJECT_ROOT))
        except ValueError:
            task["initial_frame"] = str(resolved)  # outside the repo → absolute
        tasks.append(task)

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for task in tasks:
            fh.write(json.dumps(task, ensure_ascii=False) + "\n")

    # Report initial-frame availability — the only external file a task needs.
    root = PROJECT_ROOT
    missing = [t["initial_frame"] for t in tasks if not (root / t["initial_frame"]).exists()]
    print(f"wrote {len(tasks)} tasks -> {out}")
    if missing:
        print("MISSING initial frames (verify paths):")
        for m in missing:
            print(f"  {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
