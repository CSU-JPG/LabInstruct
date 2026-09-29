"""LTX-2.3 image-to-video adapter — official repo.

Runs inside the LTX-2 official uv env (models.yaml ``env``). Invokes the
official two-stage TI2V pipeline CLI, which loads the distilled checkpoint +
spatial upsampler + distilled LoRA for 1080p output:

    python -m ltx_pipelines.ti2vid_two_stages
        --checkpoint-path <distilled ckpt> --distilled-lora <lora> 1.0
        --spatial-upsampler-path <upscaler> --gemma-root <gemma dir>
        --image <first frame> 0 1.0 --prompt <text> --seed <seed>
        --height <H> --width <W> --num-frames <N> --frame-rate <fps>
        --num-inference-steps <steps>
        --video-cfg-guidance-scale <guidance> --output-path <out.mp4>

LTX-2.3 is a joint audio-video generator; this benchmark keeps only the video
track.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
from typing import Any

from bench.adapters.base import BaseAdapter, GenerationResult
from bench.registry import GenerationTask

_DEFAULT_PARAMS = {
    "resolution": "1080p",
    "steps": 30,
    "guidance_scale": 3.0,
    "seed": 10,  # the LTX CLI's own default seed
}


def _fit_size(aspect: float, target: str = "1080p") -> tuple[int, int]:
    """Fit the source aspect ratio to the nearest official LTX-2 size.

    LTX-2 targets 1080p; the two-stage pipeline generates at half resolution
    and upsamples 2x. Keep the source aspect ratio, align the long edge to the
    target bracket, and scale the other edge to a multiple of 64
    (assert_resolution requires h/w divisible by 64 in the two-stage path).
    """
    max_w, max_h = {"1080p": (1920, 1080), "720p": (1280, 720), "480p": (832, 448)}.get(target, (1920, 1080))
    if aspect >= 1.0:
        w, h = max_w, max(64, int(round(max_w / aspect / 64) * 64))
    else:
        h, w = max_h, max(64, int(round(max_h * aspect / 64) * 64))
    return w, h


class LTXAdapter(BaseAdapter):
    model_id = "ltx2.3"

    def __init__(self, model_cfg: dict[str, Any], root: pathlib.Path):
        super().__init__(model_cfg, root)
        self.params = {**_DEFAULT_PARAMS, **model_cfg.get("params", {})}
        self.repo = pathlib.Path(self.params.get("repo", ""))

    def resolution(self) -> str:
        """Generation resolution.

        models.yaml's top-level ``resolution`` (the setting reported in the
        paper's Table 3) wins; ``params.resolution`` overrides it when present.
        """
        return str(self.cfg.get("resolution") or self.params.get("resolution") or "1080p")

    def available(self) -> str | None:
        if not self.repo.exists():
            return f"LTX-2 repo not found: {self.repo}"
        pkg = self.repo / "packages" / "ltx-pipelines" / "src" / "ltx_pipelines"
        if not (pkg / "ti2vid_two_stages.py").exists():
            return f"ltx_pipelines.ti2vid_two_stages not found under {self.repo}"
        for key in ("checkpoint_path", "distilled_lora", "spatial_upsampler_path", "gemma_root"):
            if not self.params.get(key):
                return f"params.{key} not set"
            if not pathlib.Path(str(self.params[key])).exists():
                return f"params.{key} not found: {self.params[key]}"
        return None

    def generate(
        self,
        task: GenerationTask,
        output_dir: pathlib.Path,
        gpu: int,
    ) -> GenerationResult:
        # Resolve first: the subprocess runs with cwd set to the LTX-2 repo, so
        # a relative --output-path would resolve there (and a missing parent
        # makes pyav raise FileNotFoundError, failing the whole pipeline).
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        meta: dict[str, Any] = {"model": self.model_id, "device": f"cuda:{gpu}"}
        try:
            size = _fit_size(task.aspect_ratio, self.resolution())
            num_frames = self.frames_for(task)
            seed = int(self.params.get("seed", 10))
            steps = int(self.params.get("steps", 30))
            out = output_dir / "video.mp4"
            cmd = [
                sys.executable, "-m", "ltx_pipelines.ti2vid_two_stages",
                "--checkpoint-path", str(self.params["checkpoint_path"]),
                "--distilled-lora", str(self.params["distilled_lora"]), "1.0",
                "--spatial-upsampler-path", str(self.params["spatial_upsampler_path"]),
                "--gemma-root", str(self.params["gemma_root"]),
                "--image", str(task.initial_frame), "0", "1.0",
                "--prompt", task.prompt_for_gen,
                "--seed", str(seed),
                "--height", str(size[1]),
                "--width", str(size[0]),
                "--num-frames", str(num_frames),
                "--frame-rate", str(self.fps_for(task)),
                "--num-inference-steps", str(steps),
                "--video-cfg-guidance-scale", str(self.params.get("guidance_scale", 3.0)),
                "--output-path", str(out),
            ]
            try:
                subprocess.run(cmd, cwd=self.repo, check=True, capture_output=True,
                               text=True, timeout=60 * 60 * 3)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"ti2vid_two_stages failed (rc={exc.returncode}): {exc.stderr[-2000:]}"
                ) from exc
            if not out.exists():
                raise FileNotFoundError(f"ti2vid_two_stages produced no {out}")
            meta.update(
                {
                    "size": list(size),
                    "frames": num_frames,
                    "fps": self.fps_for(task),
                    "steps": steps,
                    "guidance_scale": self.params.get("guidance_scale", 3.0),
                    "seed": seed,
                    "checkpoint_path": str(self.params["checkpoint_path"]),
                    "distilled_lora": str(self.params["distilled_lora"]),
                    "spatial_upsampler_path": str(self.params["spatial_upsampler_path"]),
                    "gemma_root": str(self.params["gemma_root"]),
                }
            )
            self._write_meta(output_dir, meta)
            return GenerationResult(video_path=out, meta=meta)
        except Exception as exc:  # noqa: BLE001 - surface adapter errors as results
            meta["error"] = repr(exc)
            self._write_meta(output_dir, meta)
            return GenerationResult(video_path=None, meta=meta, error=repr(exc))
