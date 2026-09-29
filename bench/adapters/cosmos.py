"""Cosmos3 (Nano / Super) adapter — NVIDIA official framework.

The diffusers route does NOT work for Cosmos3: the released checkpoints
(nvidia/Cosmos3-Nano / -Super) declare ``Cosmos3OmniDiffusersPipeline``,
which is not shipped by the diffusers versions currently pinned in the
cosmos3 repo. The correct path is NVIDIA's own inference framework, in the
checkout that models.yaml points at (its env already has the framework + torch):

    OmniInference.create(OmniSetupOverrides(checkpoint_path="Cosmos3-Nano"))
    -> OmniSampleOverrides(prompt, vision_path=<first frame>, num_frames,
                           fps, resolution, aspect_ratio, seed)
    -> get_sample_data -> pipe.generate_batch -> output_dir/vision.mp4

``vision_path`` provides the first-frame image conditioning; ``num_frames > 1``
selects image-to-video generation (the VFM modality is inferred from
``vision_path`` + ``num_frames``, see cosmos_framework/inference/args.py).
"""
from __future__ import annotations

import os
import pathlib
import shutil
from typing import Any

from bench.adapters.base import BaseAdapter, GenerationResult
from bench.registry import GenerationTask

_CHECKPOINT_BY_MODEL = {
    "cosmos3-nano": "Cosmos3-Nano",
    "cosmos3-super": "Cosmos3-Super",
}
_DEFAULT_PARAMS = {
    "resolution": "480",
    "fps": 24,
    # Sampling defaults = the framework's image2video modality defaults,
    # verified at
    # cosmos_framework/inference/defaults/image2video/sample_args.json
    # (num_steps=35, guidance=6.0, shift=10.0). These are shared by BOTH
    # Cosmos3-Nano and Cosmos3-Super — the per-model YAML configs
    # (configs/model/Cosmos3-Nano.yaml, -Super.yaml) contain no sampling
    # section. Values here are the fallback; models.yaml#params is the
    # per-model source of truth.
    "steps": 35,
    "guidance_scale": 6.0,
    "shift": 10.0,
    "seed": 0,
    # Prompt upsampling is OFF by default so the benchmark feeds the raw
    # instruction to the model verbatim (reproducible). Set True to engage the
    # in-model V4.2 upsampler (reasoner LM) for ablation studies.
    "prompt_upsampling": False,
}


def _nearest_aspect_ratio(aspect: float) -> str:
    """Map the source aspect ratio to the framework's allowed set."""
    targets = {"1,1": 1.0, "4,3": 4 / 3, "3,4": 3 / 4, "16,9": 16 / 9, "9,16": 9 / 16}
    return min(targets, key=lambda k: abs(targets[k] - aspect))


class CosmosAdapter(BaseAdapter):
    model_id = "cosmos3-nano"  # registry key; super is handled via cfg below

    def __init__(self, model_cfg: dict[str, Any], root: pathlib.Path):
        super().__init__(model_cfg, root)
        self.params = {**_DEFAULT_PARAMS, **model_cfg.get("params", {})}

    def available(self) -> str | None:
        try:
            import cosmos_framework  # noqa: F401
        except ImportError:
            return "cosmos_framework not importable — run in the Cosmos env (models.yaml env)"
        return None

    def _is_rank0(self) -> bool:
        """True for single-process runs or the launcher rank under torchrun.

        Uses the ``RANK`` env var set by torchrun (not torch.distributed
        state): this check runs both before and after the framework initializes
        the process group, so ``dist.is_initialized()`` is False on every rank
        at adapter entry and would make all ranks act as rank 0.
        """
        return os.environ.get("RANK", "0") == "0"

    def generate(
        self,
        task: GenerationTask,
        output_dir: pathlib.Path,
        gpu: int,
    ) -> GenerationResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        meta: dict[str, Any] = {"model": self.model_id, "device": f"cuda:{gpu}"}
        rank0 = self._is_rank0()
        try:
            self._run_framework(task, output_dir, meta)
            out = output_dir / "video.mp4"
            if rank0 and not out.exists():
                raise FileNotFoundError(f"framework produced no {out}")
            if rank0:
                self._write_meta(output_dir, meta)
            return GenerationResult(video_path=out, meta=meta)
        except Exception as exc:  # noqa: BLE001
            if rank0:
                meta["error"] = repr(exc)
                self._write_meta(output_dir, meta)
            return GenerationResult(video_path=None, meta=meta, error=repr(exc))

    def _run_framework(
        self, task: GenerationTask, output_dir: pathlib.Path, meta: dict[str, Any]
    ) -> None:
        from cosmos_framework.inference.args import (
            OmniSampleOverrides,
            OmniSetupOverrides,
        )
        from cosmos_framework.inference.common.init import init_script
        from cosmos_framework.inference.inference import OmniInference, get_sample_data

        checkpoint = _CHECKPOINT_BY_MODEL.get(self.model_id, "Cosmos3-Nano")
        resolution = str(self.params.get("resolution", "480"))
        num_frames = self.frames_for(task)
        fps = self.fps_for(task)
        aspect = _nearest_aspect_ratio(task.aspect_ratio)

        init_script()
        setup_args = OmniSetupOverrides(
            checkpoint_path=checkpoint,
            output_dir=str(output_dir),
            # throughput: pure FSDP weight sharding (cp=cfgp=1); on multi-GPU
            # torchrun launches dp_shard_size defaults to WORLD_SIZE, which is
            # exactly the paper's 4-GPU Super recipe.
            parallelism_preset="throughput",
            # Disable the framework's content-safety guardrail: its exact-match
            # blocklist rejects innocuous lab prompts (e.g. the "Vortex Genie 2"
            # mixer brand hits the blocklist word "Genie"), so it cannot gate a
            # benchmark run. Also skips the video face-blur postprocessor.
            guardrails=False,
        ).build_setup()
        pipe = OmniInference.create(setup_args)

        sample_args = OmniSampleOverrides(
            name=task.task_id,
            output_dir=str(output_dir),
            prompt=task.prompt_for_gen,
            vision_path=str(task.initial_frame),
            num_frames=num_frames,
            fps=fps,
            resolution=resolution,
            aspect_ratio=aspect,
            seed=int(self.params.get("seed", 0)),
            num_steps=int(self.params.get("steps", 35)),
            guidance=float(self.params.get("guidance_scale", 6.0)),
            shift=float(self.params.get("shift", 10.0)),
            prompt_upsampling=bool(self.params.get("prompt_upsampling", False)),
        ).build_sample(model_config=pipe.model_config)

        data_batch = get_sample_data(sample_args, model=pipe.model)
        pipe.generate_batch([sample_args], data_batch)

        # generate_batch writes <output_dir>/vision.mp4 (vision_extension).
        # File handling happens on rank 0 only; other ranks return after the
        # collective call so the framework's distributed workers finish.
        if not self._is_rank0():
            return
        produced = next(output_dir.glob("vision.*"), None)
        if produced is None:
            raise FileNotFoundError(f"no vision.* output in {output_dir}")
        shutil.move(str(produced), str(output_dir / "video.mp4"))

        meta.update(
            {
                "checkpoint": checkpoint,
                "size": f"{resolution}p",
                "aspect_ratio": aspect,
                "frames": num_frames,
                "fps": fps,
                "seed": int(self.params.get("seed", 0)),
                "num_steps": int(self.params.get("steps", 35)),
                "guidance": float(self.params.get("guidance_scale", 6.0)),
                "shift": float(self.params.get("shift", 10.0)),
                "prompt_upsampling": bool(self.params.get("prompt_upsampling", False)),
            }
        )
