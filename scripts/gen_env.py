#!/usr/bin/env python
"""Subprocess entry point that runs inside a model's environment.

Usage (invoked by bench/core/runner.py, do not call directly):

    python scripts/gen_env.py --model wan2.2 \
        --task /abs/runs/NAME/tasks/t.json \
        --out  /abs/runs/NAME/gen/TASK/wan2.2 \
        --gpu 0 --root /abs/path/to/project

Prints exactly one JSON line on stdout, prefixed with ``RESULT_JSON:`` (see
bench/core/runner.py): the GenerationResult record.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--gpus", default=None, help="comma-separated GPU indices (multi-GPU torchrun launch)")
    ap.add_argument("--root", required=True, type=pathlib.Path)
    args = ap.parse_args()

    root: pathlib.Path = args.root.resolve()
    sys.path.insert(0, str(root))

    # Pin the assigned GPU before torch/framework get imported so adapters
    # that load heavy CUDA stacks (Cosmos3 framework, diffusers) see only this
    # GPU. Under torchrun (--gpus set) CUDA_VISIBLE_DEVICES is already set by
    # the launcher and must not be overridden.
    if not args.gpus:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    # Multi-GPU: every rank runs this script; only rank 0 emits the result
    # record so the runner parses exactly one JSON line. The non-launching
    # ranks still execute adapter.generate() — the framework drives the
    # distributed compute and the adapter skips file writes off rank 0.
    # RANK is set by torchrun and is reliable before torch.distributed inits.
    is_rank0 = os.environ.get("RANK", "0") == "0"

    from bench.adapters import build_adapter
    from bench.core.runner import RESULT_MARKER
    from bench.registry import load_models, task_from_dict

    with open(args.task, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    task = task_from_dict(raw, root)

    model_cfg = load_models()[args.model]
    adapter = build_adapter(args.model, model_cfg, root)
    unavailable = adapter.available()
    if unavailable:
        record = {
            "status": "error",
            "video": None,
            "meta": {"model": args.model},
            "error": f"adapter unavailable: {unavailable}",
        }
        if is_rank0:
            print(f"{RESULT_MARKER}{json.dumps(record, ensure_ascii=False)}")
        return 1

    result = adapter.generate(task, args.out, args.gpu)
    record = {
        "status": "ok" if result.ok else "error",
        "video": str(result.video_path) if result.video_path else None,
        "meta": result.meta,
        "error": result.error,
    }
    if is_rank0:
        print(f"{RESULT_MARKER}{json.dumps(record, ensure_ascii=False)}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(_main())
