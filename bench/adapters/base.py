"""Adapter contract.

Every generator is wrapped in a ``BaseAdapter`` subclass that implements the
same ``generate()`` signature. The dispatcher does not care which model or
which environment it is talking to — the I/O contract is identical:

    generate(task, output_dir, gpu) -> GenerationResult

The adapter itself runs either
  * in-process (``run: api`` / ``run: local``), or
  * as a subprocess inside the model's own environment (``run: env``),
    launched from scripts/gen_env.py.
"""
from __future__ import annotations

import abc
import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any, Optional

from bench.registry import GenerationTask

logger = logging.getLogger("labinstruct")


@dataclass
class GenerationResult:
    video_path: Optional[pathlib.Path]
    meta: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.video_path is not None


class BaseAdapter(abc.ABC):
    """Interface implemented by every model wrapper."""

    #: adapter key in bench/models.yaml (e.g. "wan2.2")
    model_id: str = "base"

    def __init__(self, model_cfg: dict[str, Any], root: pathlib.Path):
        self.cfg = model_cfg
        self.root = root  # LabInstruct project root (for resolving relative paths)

    def available(self) -> Optional[str]:
        """Return None if the adapter can run here, else a reason string.

        Used by the dispatcher to skip models whose dependencies are missing
        in a shared env (Route B) or whose API key is unset.
        """
        return None

    @abc.abstractmethod
    def generate(
        self,
        task: GenerationTask,
        output_dir: pathlib.Path,
        gpu: int,
    ) -> GenerationResult:
        """Write the generated video into ``output_dir`` and return its path."""
        raise NotImplementedError

    # ------------------------------------------------------------------ gen params
    # Generation settings (frames / fps / duration) are PER-MODEL — the paper
    # Table 6 values recorded in models.yaml (l1_frames / l2_frames / fps) —
    # not per-task. Tasks only carry the level; adapters look up their own
    # frame counts / fps here instead of reading task.frames/task.fps.

    def frames_for(self, task: GenerationTask) -> int:
        """Frame count for this (model, task), from models.yaml (Table 6)."""
        key = "l1_frames" if task.level == 1 else "l2_frames"
        return int(self.cfg.get(key) or 96)

    def fps_for(self, task: GenerationTask) -> int:
        """FPS for this model, from models.yaml."""
        return int(self.cfg.get("fps") or 24)

    def duration_for(self, task: GenerationTask) -> float:
        """Target duration (s) derived from the model's frames/fps."""
        return round(self.frames_for(task) / self.fps_for(task), 3)

    # ------------------------------------------------------------------ utils
    def _write_meta(self, output_dir: pathlib.Path, meta: dict[str, Any]) -> pathlib.Path:
        meta_path = output_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        return meta_path
