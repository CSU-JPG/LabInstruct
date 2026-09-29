"""MiniMax H3 image-to-video adapter (Modular diffusers, fl2va workflow).

MiniMax-H3 is an omni-modal model whose bf16 weights are ~134 GB (61.7 GB
transformer + 62.1 GB Qwen3-VL conditioner + video/audio VAEs). It is served
only through the experimental Modular Diffusers API — the classic
``DiffusionPipeline`` does not support it, so this adapter builds its own
pipeline on top of ``BaseAdapter`` rather than reusing a shared one.

Because nothing fits on a single accelerator, the components are registered in
a ``ComponentsManager`` and auto-CPU-offloaded: the weights stay in host RAM
and the manager moves each component onto the GPU only for the step that needs
it (memory_reserve_margin leaves headroom for activations).

This adapter runs inside the MiniMax-H3 venv (models.yaml ``run: env``, with
``env`` pointing at that checkout's own ``.venv/bin/python``).

Frame counts are model-constrained: ``num_frames`` must be 17n+5 (video VAE
clip_length=17) AND land in [120, 360] (5-15 s at 24 fps). The values in
models.yaml are already valid, so the alignment below is a no-op for them.
"""
from __future__ import annotations

import os
import pathlib
from typing import Any, Optional

from bench.adapters.base import BaseAdapter, GenerationResult, logger
from bench.registry import GenerationTask

# num_frames must be 17n+5 (video VAE clip_length=17) AND land in [120, 360]
# (5-15 s at 24 fps). The first valid value is 124; the last is 345.
_FRAME_MIN = 124
_FRAME_MAX = 345


def _align_frames(num_frames: int) -> int:
    while num_frames % 17 != 5:
        num_frames += 1
    if num_frames < _FRAME_MIN:
        num_frames = _FRAME_MIN
    if num_frames > _FRAME_MAX:
        raise ValueError(
            f"num_frames {num_frames} exceeds MiniMax-H3's 360-frame (15 s @ 24 fps) limit"
        )
    return num_frames


def _fit_canvas(image_size: tuple[int, int], short_edge: int = 768) -> tuple[int, int]:
    """Canvas with a ``short_edge``-pixel short side, sides rounded to 32 px.

    MiniMax-H3 defaults its canvas to the first keyframe's aspect ratio with a
    768 px short edge, and height/width must be multiples of 32.
    """
    w, h = image_size
    aspect = w / h
    if aspect >= 1.0:
        return (max(32, round(short_edge * aspect / 32) * 32), short_edge)
    return (short_edge, max(32, round(short_edge / aspect / 32) * 32))


def _save_video(frames, path: pathlib.Path, fps: int = 24) -> None:
    """Write PIL-frame list (or a tensor) to an MP4, tolerant of a batch dim."""
    import imageio.v2 as imageio
    import numpy as np

    import torch

    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().numpy()
    arr = np.asarray(frames)
    if arr.ndim == 5 and arr.shape[0] == 1:  # (1,T,H,W,C)
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[1] in (1, 3) and arr.shape[1] != arr.shape[3]:
        arr = np.transpose(arr, (0, 2, 3, 1))  # (T,C,H,W) -> (T,H,W,C)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=1) as w:
        for f in arr:
            w.append_data(f)


def _save_audio(audio, path: pathlib.Path, sample_rate: int) -> None:
    """Write the audio-VAE output to a WAV, normalizing (B,C,S)/(C,S)->(S,C)."""
    import numpy as np
    import soundfile as sf

    import torch

    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().float().numpy()
    audio = np.asarray(audio)
    while audio.ndim > 2 and audio.shape[0] == 1:  # (1,C,S)->(C,S)
        audio = audio[0]
    if audio.ndim == 2 and audio.shape[0] < audio.shape[1]:  # (C,S)->(S,C)
        audio = audio.T
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.ascontiguousarray(audio), sample_rate)


class MiniMaxAdapter(BaseAdapter):
    model_id = "minimax-h3"
    default_resolution = "768p"
    default_params = {
        "steps": 30,
    }

    def __init__(self, model_cfg: dict[str, Any], root: pathlib.Path):
        super().__init__(model_cfg, root)
        self.params = {**self.default_params, **model_cfg.get("params", {})}

    def available(self) -> Optional[str]:
        try:
            from diffusers import ModularPipeline  # noqa: F401
        except ImportError:
            return "Modular diffusers (diffusers>=0.40 dev) not installed in this env"
        if not self.cfg.get("hf_id"):
            return "hf_id not set in models.yaml"
        return None

    def frames_for(self, task: GenerationTask) -> int:
        """Model frames: aligned to 17n+5 and clamped to [120, 360]."""
        return _align_frames(super().frames_for(task))

    def _exec_device(self, gpu: int) -> str:
        """Actual torch device.

        Under scripts/gen_env.py the assigned physical GPU is pinned via
        ``CUDA_VISIBLE_DEVICES``, so the only visible device is cuda:0; a
        standalone run keeps physical indices.
        """
        if gpu < 0:
            return "cpu"
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            return "cuda:0"
        return f"cuda:{gpu}"

    def generate(
        self,
        task: GenerationTask,
        output_dir: pathlib.Path,
        gpu: int,
    ) -> GenerationResult:
        device = self._exec_device(gpu)
        meta: dict[str, Any] = {
            "model": self.model_id,
            "device": f"cuda:{gpu}" if gpu >= 0 else "cpu",  # physical GPU label
        }
        try:
            import torch
            from PIL import Image
            from diffusers import ComponentsManager, ModularPipeline

            hf_id = self.cfg["hf_id"]
            workflow = self.params.get("workflow", "fl2va")
            manager = ComponentsManager()
            # workflow goes to from_pretrained only: passing it to
            # load_components as well prunes the blocks and raises a
            # _workflow_map error.
            pipe = ModularPipeline.from_pretrained(
                hf_id, workflow=workflow, components_manager=manager
            )
            pipe.load_components(dtype=torch.bfloat16)
            manager.enable_auto_cpu_offload(
                device=device,
                memory_reserve_margin=self.params.get("memory_reserve_margin", "12GB"),
            )

            with Image.open(task.initial_frame) as img:
                image = img.convert("RGB")
            size = _fit_canvas(image.size, short_edge=int(self.params.get("short_edge", 768)))
            image = image.resize(size)
            meta["size"] = list(size)

            num_frames = self.frames_for(task)
            steps = int(self.params.get("steps", 30))
            seed = int(self.params.get("seed", 0))
            fps = self.fps_for(task)
            with_audio = bool(self.params.get("with_audio", True))
            outputs = ["videos"] + (["audio", "sampling_rate"] if with_audio else [])
            meta.update(
                {
                    "seed": seed,
                    "steps": steps,
                    "frames": num_frames,
                    "fps": fps,
                    "hf_id": hf_id,
                    "workflow": workflow,
                    "with_audio": with_audio,
                }
            )

            results = pipe(
                prompt=task.prompt_for_gen,
                image=image,
                num_frames=num_frames,
                num_inference_steps=steps,
                height=size[1],
                width=size[0],
                generator=torch.Generator().manual_seed(seed),
                output=outputs,
            )

            out = output_dir / "video.mp4"
            _save_video(results["videos"][0], out, fps)

            if with_audio:
                audio = results.get("audio")
                if isinstance(audio, (list, tuple)):
                    audio = audio[0]
                if audio is not None and getattr(audio, "numel", lambda: 0)() > 0:
                    wav = output_dir / "audio.wav"
                    sr = int(results["sampling_rate"])
                    _save_audio(audio, wav, sr)
                    meta["audio"] = str(wav)
                    meta["sample_rate"] = sr

            self._write_meta(output_dir, meta)
            return GenerationResult(video_path=out, meta=meta)
        except Exception as exc:  # noqa: BLE001 - surface adapter errors as results
            logger.exception("adapter %s failed on %s", self.model_id, task.task_id)
            meta["error"] = repr(exc)
            self._write_meta(output_dir, meta)
            return GenerationResult(video_path=None, meta=meta, error=repr(exc))
