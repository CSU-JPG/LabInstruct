"""LingBot-Video image-to-video adapter (official repo).

Grounded in the model's own checkout (models.yaml ``params.repo``):

  * DiT inference   : ``scripts/inference.py --backend diffusers --mode ti2v``
  * Prompt rewriting: ``rewriter/inference.py --backend transformers --mode ti2v``
    LingBot's DiT does NOT accept raw text — a structured JSON prompt is
    required (docs/en/dit_inference.md). This adapter rewrites the paper
    instruction into the structured prompt, then runs the official runner.

This adapter runs *inside* the LingBot env (models.yaml ``env``), so
``sys.executable`` is the repo's own Python 3.11 and the subprocesses below
resolve correctly.
"""
from __future__ import annotations

import os
import pathlib
import socket
import subprocess
import sys
from typing import Any

from bench.adapters.base import BaseAdapter, GenerationResult
from bench.registry import GenerationTask

_DEFAULT_PARAMS = {
    "steps": 40,
    "guidance_scale": 3.0,
    "shift": 3.0,
    "run_refiner": False,
    "resolution": "480p",
    "rewriter_duration": None,  # defaults to model duration (BaseAdapter.duration_for)
}


def _fit_size(aspect: float, target: str = "480p") -> tuple[int, int]:
    max_edge = {"1080p": 1920, "768p": 1366, "720p": 1280, "480p": 854}.get(target, 854)
    # Both dims must be multiples of 16 (LingBot pipeline requirement); the
    # nominal max edge (e.g. 854) is not, so snap it down to a 16 multiple.
    max_edge = max(16, max_edge - (max_edge % 16))
    if aspect >= 1.0:
        return max_edge, max(16, int(round(max_edge / aspect / 16) * 16))
    return max(16, int(round(max_edge * aspect / 16) * 16)), max_edge


def _nearest_4n1(frames: int) -> int:
    """Smallest 4n+1 >= frames; LingBot requires num_frames in {1, 4n+1}."""
    if frames < 5:
        return 1
    return ((frames + 2) // 4) * 4 + 1


def _free_port() -> int:
    """Pick an unused TCP port for the nested torchrun rendezvous.

    Bench itself already runs under an (outer) torchrun; the inner launcher
    must not reuse that port or the fixed 29500 default, which other cluster
    jobs may hold. bind(0) asks the kernel for a free ephemeral port (tiny
    TOCTOU race between close and the inner launch — acceptable here).
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LingBotAdapter(BaseAdapter):
    model_id = "lingbot-video"

    def __init__(self, model_cfg: dict[str, Any], root: pathlib.Path):
        super().__init__(model_cfg, root)
        self.params = {**_DEFAULT_PARAMS, **model_cfg.get("params", {})}
        self.repo = pathlib.Path(self.params.get("repo") or self.params.get("model_dir"))

    def available(self) -> str | None:
        if not (self.repo / "scripts" / "inference.py").exists():
            return f"lingbot repo scripts not found under {self.repo}"
        model_dir = self.params.get("model_dir")
        if model_dir and not pathlib.Path(model_dir).exists():
            return f"lingbot model_dir not found: {model_dir}"
        return None

    # ------------------------------------------------------------------ utils
    def _run(self, cmd: list[str], env: dict[str, str] | None = None) -> None:
        """Run a subprocess and, on failure, raise with its captured output.

        Previously ``check=True`` swallowed stdout/stderr, so every failure was
        recorded only as ``CalledProcessError(1, [...])`` with no traceback.
        """
        proc = subprocess.run(cmd, cwd=self.repo, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(
                "command failed (rc=%s): %s\n--- stdout ---\n%s\n--- stderr ---\n%s"
                % (proc.returncode, subprocess.list2cmdline(cmd), proc.stdout, proc.stderr)
            )

    def _build_prompt_json(
        self, task: GenerationTask, output_dir: pathlib.Path
    ) -> pathlib.Path:
        """Rewrite the raw instruction into LingBot's structured JSON prompt."""
        out = output_dir / "prompt.json"
        duration = self.params.get("rewriter_duration") or self.duration_for(task)
        # The rewriter defaults to the LingBot venv (sys.executable), which
        # lacks causal-conv1d: hybrid linear attention falls back to torch and
        # decodes at ~2.2 tok/s. models.yaml can point ``rewriter_python`` at
        # an env that has the kernels (~4x faster). See the lingbot-video
        # entry in models.yaml.
        python_bin = str(self.params.get("rewriter_python") or sys.executable)
        cmd = [
            python_bin,
            "rewriter/inference.py",
            "--backend", "transformers",
            "--mode", "ti2v",
            "--prompt", task.prompt_for_gen,
            "--first-frame", str(task.initial_frame),
            "--duration", str(duration),
            "--output", str(out),
            "--base", str(self.params.get("rewriter_base_model", "")),
            "--adapter", str(self.params.get("rewriter_adapter", "")),
        ]
        self._run(cmd)
        return out

    def _run_runner(
        self,
        task: GenerationTask,
        prompt_json: pathlib.Path,
        out: pathlib.Path,
        size: tuple[int, int],
        num_frames: int,
    ) -> None:
        cmd = [
            sys.executable,
            "scripts/inference.py",
            "--backend", "diffusers",
            "--mode", "ti2v",
            "--model_dir", str(self.params["model_dir"]),
            "--image", str(task.initial_frame),
            "--prompt_json", str(prompt_json),
            "--output", str(out),
            "--height", str(size[1]),
            "--width", str(size[0]),
            "--num_frames", str(num_frames),
            "--fps", str(self.fps_for(task)),
            "--steps", str(self.params["steps"]),
            "--guidance_scale", str(self.params["guidance_scale"]),
            "--shift", str(self.params["shift"]),
        ]
        if self.params.get("run_refiner"):
            cmd += ["--run_refiner"]
        self._run(cmd)

    # ------------------------------------------------------------------ api
    def generate(
        self,
        task: GenerationTask,
        output_dir: pathlib.Path,
        gpu: int,
    ) -> GenerationResult:
        # Resolve first: the subprocesses (rewriter / inference.py) run with
        # cwd set to the LingBot repo, so a relative --output would resolve
        # there (and a missing parent makes the write fail). An absolute path
        # guarantees prompt.json / video.mp4 land in output_dir itself.
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        meta: dict[str, Any] = {"model": self.model_id, "device": f"cuda:{gpu}"}

        # Multi-GPU (models.yaml num_gpus > 1): the bench runner starts N
        # gen_env ranks under an outer torchrun and every rank calls
        # generate(). Only RANK==0 actually spawns the official torchrun below
        # (FSDP sharding + context parallelism; see _run_distributed_runner);
        # the other ranks return ok immediately so gen_env exits 0 -- the outer
        # torchrun fails the whole group if any single rank exits non-zero.
        world_size = int(os.environ.get("WORLD_SIZE") or "1")
        rank = int(os.environ.get("RANK") or "0")
        multi_gpu = world_size > 1
        if multi_gpu:
            meta["multi_gpu"] = True
            meta["world_size"] = world_size
        if multi_gpu and rank != 0:
            meta["rank"] = rank
            return GenerationResult(video_path=output_dir / "video.mp4", meta=meta)
        if multi_gpu:
            meta["device"] = "cuda:" + ",".join(str(g) for g in self._gpu_mask())

        try:
            size = _fit_size(task.aspect_ratio, self.params["resolution"])
            num_frames = _nearest_4n1(self.frames_for(task))
            prompt_json = self._build_prompt_json(task, output_dir)
            out = output_dir / "video.mp4"
            if multi_gpu:
                self._run_distributed_runner(task, prompt_json, out, size, num_frames)
            else:
                self._run_runner(task, prompt_json, out, size, num_frames)
            meta.update(
                {
                    "size": list(size),
                    "frames": num_frames,
                    "fps": self.fps_for(task),
                    "steps": self.params["steps"],
                    "guidance_scale": self.params["guidance_scale"],
                    "shift": self.params["shift"],
                    "run_refiner": bool(self.params.get("run_refiner")),
                }
            )
            if multi_gpu:
                meta["cp_degree"] = len(self._gpu_mask())
            self._write_meta(output_dir, meta)
            return GenerationResult(video_path=out, meta=meta)
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            meta["error"] = err
            self._write_meta(output_dir, meta)
            return GenerationResult(video_path=None, meta=meta, error=err)

    # ------------------------------------------------------------------ multi-GPU
    def _gpu_mask(self) -> list[int]:
        """Physical GPUs this job owns.

        The bench runner reserved ``num_gpus`` cards and set
        ``CUDA_VISIBLE_DEVICES`` to that mask before launching gen_env; under
        torchrun every rank sees the same mask. Fall back to ``num_gpus`` only
        if the env var is absent (never the case through the bench runner).
        """
        raw = os.environ.get("CUDA_VISIBLE_DEVICES")
        if raw:
            return [int(x) for x in raw.split(",") if x.strip()]
        return list(range(int(self.cfg.get("num_gpus", 1) or 1)))

    def _run_distributed_runner(
        self,
        task: GenerationTask,
        prompt_json: pathlib.Path,
        out: pathlib.Path,
        size: tuple[int, int],
        num_frames: int,
    ) -> None:
        """Official multi-GPU run: a *nested* top-level torchrun of
        ``scripts/inference.py`` with FSDP weight sharding + context
        parallelism (docs/en/dit_inference.md; mirrors
        scripts/multi-gpus-no-refiner/run_moe_ti2v_fsdp_cp8.sh with CP=world).

        LingBot's official distributed entry point is torchrun running
        inference.py directly as the rank process — it offers no reusable
        in-process distributed API, so unlike cosmos (which runs a collective
        on every bench rank) only bench RANK==0 spawns the real work here. The
        bench runner already masked CUDA_VISIBLE_DEVICES to our GPUs; LingBot
        binds devices purely by LOCAL_RANK, so under the mask its inner ranks
        land on exactly those cards. The outer rank vars are scrubbed and a
        fresh MASTER_PORT is used so the inner launcher owns its rendezvous
        (clean external-harness isolation per the official doc).
        """
        nproc = len(self._gpu_mask())
        master_port = _free_port()
        cmd = [
            sys.executable,
            "-m", "torch.distributed.run",
            "--standalone",
            "--nproc-per-node", str(nproc),
            "--master-port", str(master_port),
            "scripts/inference.py",
            "--backend", "diffusers",
            "--mode", "ti2v",
            "--model_dir", str(self.params["model_dir"]),
            "--image", str(task.initial_frame),
            "--prompt_json", str(prompt_json),
            "--output", str(out),
            "--height", str(size[1]),
            "--width", str(size[0]),
            "--num_frames", str(num_frames),
            "--fps", str(self.fps_for(task)),
            "--steps", str(self.params["steps"]),
            "--guidance_scale", str(self.params["guidance_scale"]),
            "--shift", str(self.params["shift"]),
            # official no-refiner multi-GPU recipe: FSDP shard the base DiT and
            # context-parallel the sequence over the whole world (CP == nproc;
            # cfg_parallel stays 1 so --batch_cfg is legal, runner.py:1100).
            "--batch_cfg",
            "--enable_fsdp_inference",
            "--context_parallel_degree", str(nproc),
            "--context_parallel_ulysses_anything",
        ]
        if self.params.get("run_refiner"):
            cmd += ["--run_refiner"]
        self._run(cmd, env=self._isolated_inner_env(master_port))

    def _isolated_inner_env(self, master_port: int) -> dict[str, str]:
        """Child env for the nested torchrun.

        Scrub the bench (outer) torchrun's distributed vars so the inner
        launcher owns rendezvous and each inner rank re-reads RANK/WORLD_SIZE
        from it; set a fresh MASTER_PORT; add the official LingBot env vars
        (same ones the official scripts export). CUDA_VISIBLE_DEVICES is
        deliberately kept so inner ranks bind to the GPUs bench reserved.
        """
        env = dict(os.environ)
        for key in list(env):
            if key.startswith("TORCHELASTIC_") or key in {
                "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE",
                "MASTER_ADDR", "MASTER_PORT",
            }:
                env.pop(key, None)
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = str(master_port)
        env["DIFFUSERS_ATTN_BACKEND"] = "_native_flash"
        env["LINGBOT_MOE_PAD_BACKEND"] = "vectorized"
        env["LINGBOT_MOE_EXPERT_BACKEND"] = "grouped_mm"
        repo = str(self.repo)
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = repo + os.pathsep + str(self.repo / "rewriter") + (
            (os.pathsep + existing) if existing else ""
        )
        return env
