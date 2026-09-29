"""Subprocess runner: executes a model's adapter inside its own environment.

The unified I/O contract is a JSON task file (absolute paths) written by the
dispatcher and consumed by scripts/gen_env.py, which runs *inside* the
model's env and emits a JSON result on stdout.
"""
from __future__ import annotations

import json
import os
import pathlib
import shlex
import subprocess
import sys
from typing import Any

_GEN_ENV_SCRIPT = "scripts/gen_env.py"

# Prefix scripts/gen_env.py puts before its result record on stdout, so the
# runner never mistakes a framework JSON log line for the result.
RESULT_MARKER = "RESULT_JSON:"


def _env_to_interpreter(env: str) -> list[str]:
    """Turn models.yaml ``env`` into the token list that launches python.

    - ""                    -> current interpreter
    - "/path/.venv/bin/python3.11"  -> used as-is
    - "uv run --project /x" -> "uv run --project /x python"
    - "conda run -n wan"    -> "conda run -n wan python"
    """
    if not env.strip():
        return [sys.executable]
    tokens = shlex.split(env)
    base = tokens[-1].split("/")[-1]
    if base.startswith("python"):
        return tokens
    return tokens + ["python"]


def run_generation(
    *,
    model_id: str,
    model_cfg: dict[str, Any],
    task_json: pathlib.Path,
    out_dir: pathlib.Path,
    gpus: list[int],
    root: pathlib.Path,
    log_path: pathlib.Path,
) -> dict[str, Any]:
    """Spawn the adapter in the model's env; return the parsed result record.

    Single-GPU models run ``scripts/gen_env.py`` directly. Multi-GPU models
    (``num_gpus > 1`` in models.yaml) launch under ``torch.distributed.run`` so
    the framework sees a proper distributed world (WORLD_SIZE/RANK); only
    rank 0 emits the result record on stdout.
    """
    interpreter = _env_to_interpreter(model_cfg.get("env", ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    script = str(root / _GEN_ENV_SCRIPT)
    if len(gpus) == 1:
        cmd = interpreter + [
            script,
            "--model", model_id,
            "--task", str(task_json),
            "--out", str(out_dir),
            "--gpu", str(gpus[0]),
            "--root", str(root),
        ]
    else:
        gpu_list = ",".join(str(g) for g in gpus)
        cmd = interpreter + [
            "-m", "torch.distributed.run",
            "--nproc-per-node", str(len(gpus)),
            "--nnodes", "1",
            "--standalone",
            script,
            "--model", model_id,
            "--task", str(task_json),
            "--out", str(out_dir),
            "--gpus", gpu_list,
            "--root", str(root),
        ]
    env = dict(os.environ)
    if len(gpus) > 1:
        env["CUDA_VISIBLE_DEVICES"] = gpu_list
    proc = subprocess.run(
        cmd,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60 * 60 * 6,  # generous; 64B model loads can be slow
        env=env,
    )
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"### {model_id} {task_json.name}\n")
        fh.write(f"$ {' '.join(cmd)}\n")
        fh.write("--- stdout ---\n")
        fh.write(proc.stdout)
        fh.write("--- stderr ---\n")
        fh.write(proc.stderr)
        fh.write("\n")

    result = _extract_result(proc.stdout)
    if result is None:
        result = {
            "status": "error",
            "error": f"adapter produced no JSON result (rc={proc.returncode}); see {log_path}",
        }
    return result


def _extract_result(stdout: str) -> dict[str, Any] | None:
    """Parse the adapter's result record out of captured stdout.

    Prefers the line prefixed with RESULT_MARKER; falls back to the last line
    that looks like a JSON object (an adapter invoked directly that prints one).
    """
    lines = stdout.splitlines()
    for line in reversed(lines):
        line = line.strip()
        if line.startswith(RESULT_MARKER):
            try:
                return json.loads(line[len(RESULT_MARKER):])
            except json.JSONDecodeError:
                continue
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def result_record(task_id: str, model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "model": model_id,
        **payload,
    }
