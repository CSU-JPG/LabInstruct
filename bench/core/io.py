"""Output layout, result records, GPU query helpers."""
from __future__ import annotations

import json
import pathlib
import subprocess
from typing import Any, Iterator

# --------------------------------------------------------------------------
# Output layout
# --------------------------------------------------------------------------
# outputs/
#   {model_id}/{task_id}/video.mp4 + meta.json   (generated videos, flat)
#   runs/{run_id}/run.json + tasks/{task_id}.json + results.jsonl + dispatcher.log (metadata)
# --------------------------------------------------------------------------


def gen_out_dir(outputs_root: pathlib.Path, model_id: str, task_id: str) -> pathlib.Path:
    """Flat per-model video output: <outputs_root>/<model_id>/<task_id>/."""
    return outputs_root / model_id / task_id


def run_dir(outputs_root: pathlib.Path, run_id: str) -> pathlib.Path:
    return outputs_root / "runs" / run_id


def write_run_manifest(run_root: pathlib.Path, manifest: dict[str, Any]) -> pathlib.Path:
    run_root.mkdir(parents=True, exist_ok=True)
    path = run_root / "run.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return path


def append_result(results_jsonl: pathlib.Path, record: dict[str, Any]) -> None:
    with open(results_jsonl, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_results(results_jsonl: pathlib.Path) -> Iterator[dict[str, Any]]:
    if not results_jsonl.exists():
        return
    with open(results_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# --------------------------------------------------------------------------
# GPU helpers
# --------------------------------------------------------------------------


def gpu_free_mem_mib() -> dict[int, float]:
    """{gpu_index: free memory MiB} via nvidia-smi (no pynvml dependency)."""
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    result: dict[int, float] = {}
    if out.returncode != 0:
        return result
    for line in out.stdout.strip().splitlines():
        idx, free = line.split(",")
        result[int(idx.strip())] = float(free.strip())
    return result
