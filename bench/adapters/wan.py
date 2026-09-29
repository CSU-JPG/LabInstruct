"""Wan2.2-I2V-A14B adapter — official repo.

Runs inside the Wan2.2 official env (models.yaml ``env``). Invokes the official
inference CLI ``generate.py --task i2v-A14B`` via subprocess, so the raw
checkpoint weights (Wan-AI/Wan2.2-I2V-A14B, the ``--ckpt_dir`` layout with
VAE + T5 + high/low noise transformers) go through the maintainers' reference
path rather than the diffusers re-packaging.

    python generate.py --task i2v-A14B --ckpt_dir <ckpt> --image <first frame> \
        --size <WxH> --prompt <text> --frame_num <N> --sample_steps <steps> \
        --base_seed <seed> --save_file <out.mp4>

The official repo pins its own transformers/flash_attn versions inside its
env; this adapter only ever runs there, so it imports neither wan nor diffusers.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from typing import Any

from bench.adapters.base import BaseAdapter, GenerationResult
from bench.registry import GenerationTask

_DEFAULT_PARAMS = {
    "resolution": "720p",
    "steps": 30,
    "guidance_scale": 5.0,  # default CFG of the official wan_i2v_A14B config
    "seed": 0,
}

# generate.py's i2v-A14B task accepts only these sizes (see _validate_args /
# SUPPORTED_SIZES); the others (1280x704, 1024x704, ...) belong to t2v/ti2v.
# The paper keeps the source aspect ratio and takes the nearest official
# resolution, so pick by aspect from this pool.
_OFFICIAL_SIZES = {
    "720p": [(1280, 720), (720, 1280)],
    "480p": [(832, 480), (480, 832)],
}


def _fit_size(aspect: float, target: str = "720p") -> tuple[int, int]:
    """Snap the source aspect ratio to the nearest official Wan2.2 size."""
    pool = _OFFICIAL_SIZES.get(target, _OFFICIAL_SIZES["720p"])
    return min(pool, key=lambda wh: abs(wh[0] / wh[1] - aspect))


def _nearest_4n1(frames: int) -> int:
    """Smallest 4n+1 >= frames.

    Wan2.2 i2v requires F = 1 (mod 4): image2video.py repeats the first frame 4
    times before concatenating, giving ``F-1+4`` frames that must divide by 4
    (F=120 -> 123//4 leaves a remainder and the view shape mismatches).
    ``(frames + 2) // 4 * 4 + 1`` rounds up to 4n+1: 65 stays 65, 64 -> 65.
    """
    return ((frames + 2) // 4) * 4 + 1


class WanAdapter(BaseAdapter):
    model_id = "wan2.2"

    def __init__(self, model_cfg: dict[str, Any], root: pathlib.Path):
        super().__init__(model_cfg, root)
        self.params = {**_DEFAULT_PARAMS, **model_cfg.get("params", {})}
        self.repo = pathlib.Path(self.params.get("repo", ""))
        self.ckpt_dir = self.params.get("ckpt_dir")

    def resolution(self) -> str:
        """Generation resolution.

        models.yaml's top-level ``resolution`` (the setting reported in the
        paper's Table 3) wins; ``params.resolution`` overrides it when present.
        """
        return str(self.cfg.get("resolution") or self.params.get("resolution") or "720p")

    def _probe_video(self, path: pathlib.Path) -> dict[str, Any]:
        """Probe the produced video's real size / frame count / fps.

        Wan i2v's ``--size`` is only an area bracket: the actual output size
        follows from max_area and the first frame's aspect ratio (the 720p
        bracket yields 1104x816 for a 4:3 frame). sample_fps is fixed at 16 for
        official i2v. Record the measured values rather than the requested ones.
        """
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
                 "-of", "json", str(path)],
                capture_output=True, text=True, check=True, timeout=30)
            s = json.loads(out.stdout)["streams"][0]
            return {
                "actual_size": [int(s["width"]), int(s["height"])],
                "actual_frames": int(s.get("nb_frames", 0)),
                "actual_fps": s.get("r_frame_rate", ""),
            }
        except Exception:  # noqa: BLE001 - probe is best-effort
            return {}

    def available(self) -> str | None:
        if not self.repo.exists():
            return f"Wan2.2 repo not found: {self.repo}"
        if not (self.repo / "generate.py").exists():
            return f"Wan2.2 generate.py not found under {self.repo}"
        if not self.ckpt_dir:
            return "params.ckpt_dir (Wan2.2-I2V-A14B weights dir) not set"
        if not pathlib.Path(self.ckpt_dir).exists():
            return f"ckpt_dir not found: {self.ckpt_dir}"
        return None

    def generate(
        self,
        task: GenerationTask,
        output_dir: pathlib.Path,
        gpu: int,
    ) -> GenerationResult:
        # Resolve first: the subprocess runs with cwd set to the official repo,
        # so a relative --save_file would resolve there (and a missing parent
        # makes save_video fail silently while generate.py still exits 0). An
        # absolute path guarantees the video lands in output_dir itself.
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        meta: dict[str, Any] = {"model": self.model_id, "device": f"cuda:{gpu}"}
        try:
            size = _fit_size(task.aspect_ratio, self.resolution())
            num_frames = _nearest_4n1(self.frames_for(task))
            seed = int(self.params.get("seed", 0))
            steps = int(self.params.get("steps", 30))
            out = output_dir / "video.mp4"
            cmd = [
                sys.executable, "generate.py",
                "--task", "i2v-A14B",
                "--ckpt_dir", str(self.ckpt_dir),
                "--image", str(task.initial_frame),
                "--size", f"{size[0]}*{size[1]}",
                "--prompt", task.prompt_for_gen,
                "--frame_num", str(num_frames),
                "--sample_steps", str(steps),
                "--sample_guide_scale", str(self.params.get("guidance_scale", 5.0)),
                "--base_seed", str(seed),
                "--save_file", str(out),
            ]
            # Official memory-saving flags: 14B at 720p/121 frames peaks high
            # without flash_attn (SDPA fallback). t5_cpu moves the T5 text
            # encoder to CPU for ~10GB, convert_model_dtype lowers weight
            # precision. --offload_model is left out: on a single GPU
            # generate.py already defaults it to True.
            if self.params.get("convert_model_dtype", True):
                cmd.append("--convert_model_dtype")
            if self.params.get("t5_cpu", True):
                cmd.append("--t5_cpu")
            try:
                subprocess.run(cmd, cwd=self.repo, check=True, capture_output=True,
                               text=True, timeout=60 * 60 * 3)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"generate.py failed (rc={exc.returncode}): {exc.stderr[-2000:]}"
                ) from exc
            if not out.exists():
                raise FileNotFoundError(f"generate.py produced no {out}")
            meta.update(
                {
                    # --size is only an area bracket and sample_fps is fixed at
                    # 16 for official i2v; the requested values are kept here
                    # and the measured ones come from _probe_video below.
                    "size": list(size),  # requested bracket
                    "frames": num_frames,  # requested frame count (4n+1)
                    "fps": self.fps_for(task),  # nominal paper fps
                    "steps": steps,
                    "guidance_scale": self.params.get("guidance_scale", 5.0),
                    "seed": seed,
                    "ckpt_dir": str(self.ckpt_dir),
                    **self._probe_video(out),
                }
            )
            self._write_meta(output_dir, meta)
            return GenerationResult(video_path=out, meta=meta)
        except Exception as exc:  # noqa: BLE001 - surface adapter errors as results
            meta["error"] = repr(exc)
            self._write_meta(output_dir, meta)
            return GenerationResult(video_path=None, meta=meta, error=repr(exc))
