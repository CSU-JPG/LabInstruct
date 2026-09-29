"""Model registry + task loading.

- load_models()     -> dict[str, dict] from bench/models.yaml
- load_tasks(...)   -> list[GenerationTask] from a tasks.jsonl (or directory)
- GenerationTask    -> the unified I/O record every adapter consumes
"""
from __future__ import annotations

import json
import os
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

from bench import PROJECT_ROOT

MODELS_YAML = PROJECT_ROOT / "bench" / "models.yaml"

_ENV_VAR = re.compile(r"\$\{(\w+)\}|\$(\w+)")


def _expand_env(value: Any, unset: set[str]) -> Any:
    """Substitute ``$VAR`` / ``${VAR}`` in every string of a nested structure."""
    if isinstance(value, str):
        def substitute(match: re.Match[str]) -> str:
            name = match.group(1) or match.group(2)
            if name in os.environ:
                return os.environ[name]
            unset.add(name)
            return match.group(0)
        return _ENV_VAR.sub(substitute, value)
    if isinstance(value, dict):
        return {key: _expand_env(item, unset) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item, unset) for item in value]
    return value


def load_models(path: pathlib.Path = MODELS_YAML) -> dict[str, dict[str, Any]]:
    """Load the registry, substituting the environment variables in it.

    The repository paths and weight paths in models.yaml are ``$VAR``
    placeholders (see the file header).  A variable the file references but the
    environment does not define is an error here rather than a path that
    silently resolves to a literal ``${...}`` directory.
    """
    with open(path, "r", encoding="utf-8") as fh:
        models = yaml.safe_load(fh)
    unset: set[str] = set()
    models = _expand_env(models, unset)
    if unset:
        raise ValueError(
            f"{path} references undefined environment variable(s): "
            f"{', '.join(sorted(unset))} -- set them to your own directories"
        )
    return models


@dataclass
class GenerationTask:
    """One (initial frame, prompt_for_gen) pair -> expected video.

    Mirrors the paper's generation task (Eq. 1): G(I0, T; xi) -> V.
    ``spec`` is the structured specification S = (O, R, A, Delta, D), mirrored
    from the data/specs annotation.
    """

    task_id: str
    level: int                      # 1 = atomic (L1), 2 = procedural (L2)
    prompt_for_gen: str             # generation instruction T (spec prompt_for_gen)
    initial_frame: pathlib.Path     # I0
    # frames/fps/duration are NOT task fields — they are per-model, from
    # models.yaml (paper Table 3); adapters look them up via BaseAdapter.
    spec: dict[str, Any] = field(default_factory=dict)   # S = (O, R, A, D, ...)
    raw: dict[str, Any] = field(default_factory=dict)    # original jsonl line

    @property
    def aspect_ratio(self) -> float:
        """Width/height of the initial frame, used to keep the source AR."""
        from PIL import Image

        with Image.open(self.initial_frame) as img:
            w, h = img.size
        return w / h


def resolve_task_path(root: pathlib.Path, p: Optional[str]) -> Optional[pathlib.Path]:
    if not p:
        return None
    path = pathlib.Path(p)
    if not path.is_absolute():
        path = root / path
    return path


def load_tasks(
    tasks_source: pathlib.Path,
    root: pathlib.Path = PROJECT_ROOT,
) -> list[GenerationTask]:
    """Load tasks from a .jsonl file or from a directory of .json files.

    All relative paths inside the task records are resolved against ``root``
    so that subprocess workers (running in other envs / other cwds) see
    absolute paths.
    """
    paths = []
    if tasks_source.is_dir():
        paths = sorted(tasks_source.glob("*.json"))
        jsonl = tasks_source / "tasks.jsonl"
        if jsonl.exists():
            paths = [jsonl]
    elif tasks_source.suffix == ".jsonl":
        paths = [tasks_source]
    else:
        paths = [tasks_source]

    tasks: list[GenerationTask] = []
    for p in paths:
        if p.suffix == ".jsonl":
            with open(p, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        tasks.append(task_from_dict(json.loads(line), root))
        else:
            with open(p, "r", encoding="utf-8") as fh:
                tasks.append(task_from_dict(json.load(fh), root))
    return tasks


def task_from_dict(d: dict[str, Any], root: pathlib.Path = PROJECT_ROOT) -> GenerationTask:
    """Build a GenerationTask from a task dict (paths resolved against root).

    Public counterpart of the loader used by load_tasks(); subprocess scripts
    that run inside a model's env (scripts/gen_env.py) reuse it instead of
    re-constructing the record by hand, so the schema lives in one place.
    """
    return GenerationTask(
        task_id=d["task_id"],
        level=int(d.get("level", 1)),
        prompt_for_gen=d["prompt_for_gen"],
        initial_frame=resolve_task_path(root, d.get("initial_frame")) or pathlib.Path(),
        spec=d.get("spec", {}),
        raw=d,
    )


def save_task_json(task: GenerationTask, path: pathlib.Path) -> None:
    """Serialize a task (with absolute paths) for the subprocess contract."""
    payload = dict(task.raw)
    payload.pop("instruction", None)  # legacy key; task schema uses prompt_for_gen
    payload.update(
        {
            "task_id": task.task_id,
            "level": task.level,
            "prompt_for_gen": task.prompt_for_gen,
            "initial_frame": str(task.initial_frame),
            "spec": task.spec,
        }
    )
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
