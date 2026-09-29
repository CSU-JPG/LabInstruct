"""Dispatcher: runs task x model jobs across GPUs / envs / APIs.

Execution model:
  * ``run: api``  -> adapter runs in-process (no env, no GPU).
  * ``run: env|local`` -> adapter runs as a subprocess in the model's own
    Python env (scripts/gen_env.py), on an assigned GPU.

Concurrency: single-GPU models (``num_gpus: 1``) run in parallel, one job per
idle card, never stacked on the same card; multi-GPU models (``num_gpus > 1``,
e.g. cosmos3-super) reserve all their cards at once and therefore run alone.
API jobs (no GPU) run concurrently, capped by ``max_api_jobs``.
"""
from __future__ import annotations

import datetime
import logging
import pathlib
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

from bench import PROJECT_ROOT
from bench.core import io, runner
from bench.registry import GenerationTask, load_models, load_tasks, save_task_json

logger = logging.getLogger("labinstruct")

_DEFAULT_GPU_MEM_MIB = 143000  # fallback if nvidia-smi unavailable (4x L20X 143GB)


@dataclass
class _GPUAllocator:
    gpus: list[int]
    used_mib: dict[int, float] = field(default_factory=dict)
    capacity_mib: dict[int, float] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        # Card capacity from nvidia-smi (memory.total); fall back to default.
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False,
        )
        for gpu in self.gpus:
            self.capacity_mib[gpu] = _DEFAULT_GPU_MEM_MIB
            self.used_mib[gpu] = 0.0
        if out.returncode == 0:
            for line in out.stdout.strip().splitlines():
                idx, total = line.split(",")
                if int(idx.strip()) in self.gpus:
                    self.capacity_mib[int(idx.strip())] = float(total.strip())

    def acquire(self, model_gpu_mem_gb: float, timeout_s: float = 1e9) -> int:
        """Block until an idle card that fits the model frees up (one task per card)."""
        needed_mib = model_gpu_mem_gb * 1024
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self.lock:
                for gpu in self.gpus:
                    if self.used_mib[gpu] == 0 and needed_mib <= self.capacity_mib[gpu]:
                        self.used_mib[gpu] = needed_mib
                        return gpu
            time.sleep(5)
        raise TimeoutError("no idle GPU with enough memory became available")

    def release(self, gpu: int, model_gpu_mem_gb: float) -> None:
        with self.lock:
            self.used_mib[gpu] = max(0.0, self.used_mib[gpu] - model_gpu_mem_gb * 1024)

    def acquire_many(self, n: int, model_gpu_mem_gb: float, timeout_s: float = 1e9) -> list[int]:
        """Block until ``n`` idle cards that fit the model are free; reserve all."""
        needed_mib = model_gpu_mem_gb * 1024
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self.lock:
                free = [g for g in self.gpus if self.used_mib[g] == 0 and needed_mib <= self.capacity_mib[g]]
                if len(free) >= n:
                    picked = free[:n]
                    for g in picked:
                        self.used_mib[g] = needed_mib
                    return picked
            time.sleep(5)
        raise TimeoutError(f"no {n} idle GPUs with enough memory became available")

    def release_many(self, gpus: list[int], model_gpu_mem_gb: float) -> None:
        with self.lock:
            for g in gpus:
                self.used_mib[g] = max(0.0, self.used_mib[g] - model_gpu_mem_gb * 1024)


class Dispatcher:
    def __init__(
        self,
        *,
        models: Optional[list[str]] = None,
        task_ids: Optional[list[str]] = None,
        tasks_source: pathlib.Path = PROJECT_ROOT / "data" / "tasks.jsonl",
        run_id: Optional[str] = None,
        outputs_root: pathlib.Path = PROJECT_ROOT / "outputs",
        gpus: Optional[list[int]] = None,
        max_api_jobs: int = 4,
        continue_: bool = False,
        root: pathlib.Path = PROJECT_ROOT,
    ):
        self.root = root
        self.all_models = load_models()
        self.models = [m for m in (models or list(self.all_models)) if m in self.all_models]
        self.tasks = load_tasks(tasks_source, root=root)
        if task_ids:
            wanted = set(task_ids)
            self.tasks = [t for t in self.tasks if t.task_id in wanted]

        self.outputs_root = outputs_root
        self.run_id = run_id or datetime.datetime.now().strftime("run-%Y%m%d-%H%M%S")
        self.run_root = io.run_dir(outputs_root, self.run_id)
        self.tasks_dir = self.run_root / "tasks"
        self.results_jsonl = self.run_root / "results.jsonl"
        self.log_path = self.run_root / "dispatcher.log"
        self.gpus = gpus if gpus else list(range(len(io.gpu_free_mem_mib()))) or [0]
        self.max_api_jobs = max_api_jobs
        self.continue_ = continue_
        self.allocator = _GPUAllocator(self.gpus)

        # Progress tracking: total jobs for this run plus a thread-safe
        # completion counter (single-GPU jobs complete out of order).
        self._jobs_total = 0
        self._jobs_done = 0
        self._progress_lock = threading.Lock()

    # ------------------------------------------------------------------ jobs
    def _jobs(self) -> list[tuple[str, dict[str, Any], GenerationTask]]:
        """Build the job list.

        In continue mode a task is skipped when its video already exists
        (outputs/<model_id>/<task_id>/video.mp4). The video file is the ground
        truth for completion rather than results.jsonl, which is per-run.
        """
        jobs = []
        for model_id in self.models:
            cfg = self.all_models[model_id]
            for task in self.tasks:
                if self.continue_:
                    out_dir = io.gen_out_dir(self.outputs_root, model_id, task.task_id)
                    if (out_dir / "video.mp4").exists():
                        logger.info("skip %s/%s (video.mp4 already exists)", model_id, task.task_id)
                        continue
                jobs.append((model_id, cfg, task))
        return jobs

    # ------------------------------------------------------------------ run
    def run(self) -> pathlib.Path:
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        io.write_run_manifest(
            self.run_root,
            {
                "run_id": self.run_id,
                "models": self.models,
                "tasks": [t.task_id for t in self.tasks],
                "gpus": self.gpus,
                "created": datetime.datetime.now().isoformat(),
            },
        )
        for task in self.tasks:
            save_task_json(task, self.tasks_dir / f"{task.task_id}.json")

        jobs = self._jobs()
        self._jobs_total = len(jobs)
        logger.info("run=%s jobs=%d models=%s tasks=%d", self.run_id, len(jobs), self.models, len(self.tasks))
        if not jobs:
            logger.info("no jobs to run")
            return self.results_jsonl

        # Single-GPU jobs run concurrently, one per idle card; multi-GPU models
        # (num_gpus > 1) reserve all their cards at once and thus run alone.
        # The allocator's one-job-per-card rule means declared VRAM never stacks.
        with ThreadPoolExecutor(max_workers=len(self.gpus) + self.max_api_jobs) as pool:
            futures = [pool.submit(self._run_job, model_id, cfg, task) for model_id, cfg, task in jobs]
            for fut in futures:
                fut.result()  # results are appended inside _run_job
        return self.results_jsonl

    # ------------------------------------------------------------------ job
    def _run_job(self, model_id: str, cfg: dict[str, Any], task: GenerationTask) -> None:
        out_dir = io.gen_out_dir(self.outputs_root, model_id, task.task_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        try:
            if cfg.get("run") == "api":
                payload = self._run_api_job(model_id, cfg, task, out_dir)
            else:
                payload = self._run_env_job(model_id, cfg, task, out_dir)
            record = runner.result_record(task.task_id, model_id, payload)
        except Exception as exc:  # noqa: BLE001
            logger.exception("job %s/%s failed", task.task_id, model_id)
            record = runner.result_record(
                task.task_id, model_id, {"status": "error", "video": None, "meta": {}, "error": repr(exc)}
            )
        record["elapsed_s"] = round(time.time() - started, 1)
        io.append_result(self.results_jsonl, record)
        with self._progress_lock:
            self._jobs_done += 1
            done, total = self._jobs_done, self._jobs_total
        logger.info("[%d/%d][%s/%s] %s in %.1fs", done, total, model_id, task.task_id, record["status"], record["elapsed_s"])

    def _run_env_job(self, model_id: str, cfg: dict[str, Any], task: GenerationTask, out_dir: pathlib.Path) -> dict[str, Any]:
        gpu_mem_gb = float(cfg.get("gpu_mem_gb", 40))
        num_gpus = int(cfg.get("num_gpus", 1))
        if num_gpus > len(self.gpus):
            raise ValueError(
                f"{model_id} needs {num_gpus} GPUs but --gpus only has {len(self.gpus)}"
            )
        if num_gpus > 1:
            gpus = self.allocator.acquire_many(num_gpus, gpu_mem_gb)
        else:
            gpus = [self.allocator.acquire(gpu_mem_gb)]
        try:
            task_json = self.tasks_dir / f"{task.task_id}.json"
            payload = runner.run_generation(
                model_id=model_id,
                model_cfg=cfg,
                task_json=task_json,
                out_dir=out_dir,
                gpus=gpus,
                root=self.root,
                log_path=self.log_path,
            )
            payload["gpu"] = gpus if num_gpus > 1 else gpus[0]
            return payload
        finally:
            if num_gpus > 1:
                self.allocator.release_many(gpus, gpu_mem_gb)
            else:
                self.allocator.release(gpus[0], gpu_mem_gb)

    def _run_api_job(self, model_id: str, cfg: dict[str, Any], task: GenerationTask, out_dir: pathlib.Path) -> dict[str, Any]:
        from bench.adapters import build_adapter

        adapter = build_adapter(model_id, cfg, self.root)
        unavailable = adapter.available()
        if unavailable:
            return {"status": "error", "video": None, "meta": {}, "error": f"adapter unavailable: {unavailable}"}
        result = adapter.generate(task, out_dir, gpu=-1)
        return {
            "status": "ok" if result.ok else "error",
            "video": str(result.video_path) if result.video_path else None,
            "meta": result.meta,
            "error": result.error,
        }
