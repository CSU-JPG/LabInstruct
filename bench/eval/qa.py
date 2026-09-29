"""Prompt building, parsing and validation for atomic QA checklists.

The LLM call itself lives in ``scripts/generate_qas.py``; this module only
holds the shared vocabulary (dimensions, importance levels, item limits) and
the JSON contract the offline generators and the judges agree on.
"""
from __future__ import annotations

import json
import pathlib
import re
import warnings
from dataclasses import dataclass
from typing import Any

DIMENSIONS = [
    "Scene Consistency",
    "Object Consistency",
    "Action Fidelity",
    "State Correctness",
    "Physical Plausibility",
    "Visual Safety",
]
IMPORTANCE_LEVELS = ["critical", "standard", "supplementary"]
QA_LIMITS = {"L1": (6, 15), "L2": (10, 20)}
DEFAULT_TEMPLATE = (
    pathlib.Path(__file__).resolve().parents[2] / "data" / "rules" / "qa_generation.txt"
)


@dataclass
class QAItem:
    id: str
    dimension: str
    importance: str
    question: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "qa_id": self.id,
            "dimension": self.dimension,
            "importance": self.importance,
            "question": self.question,
        }


def task_level_from_spec_path(spec_path: pathlib.Path) -> str:
    """Read L1 or L2 from a spec filename such as ``012_L1_015_*_spec.json``."""
    match = re.search(r"(?:^|_)(L[12])(?:_|$)", spec_path.stem.upper())
    if not match:
        raise ValueError(f"cannot parse L1 or L2 from spec filename {spec_path.name!r}")
    return match.group(1)


def load_spec(spec_path: pathlib.Path) -> tuple[dict[str, Any], str]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    level = task_level_from_spec_path(spec_path)
    declared_level = str(spec.get("split", "")).upper()
    if declared_level and declared_level != level:
        raise ValueError(
            f"filename level {level} conflicts with spec split {declared_level}"
        )
    if not spec.get("task_id"):
        raise ValueError("spec must contain a non-empty task_id")
    return spec, level


def build_qa_prompt(
    spec: dict[str, Any],
    level: str,
    template_path: pathlib.Path = DEFAULT_TEMPLATE,
) -> str:
    """Render the external QA prompt template for one task's text instruction.

    The QA checklist is conditioned only on ``prompt_for_gen`` (the text
    instruction given to the video generator), plus the ``task_id``/``task_level``
    metadata required by the output format. The rest of the structured spec is
    not sent to the model.
    """
    if level not in QA_LIMITS:
        raise ValueError(f"unsupported task level {level!r}")
    task_id = str(spec.get("task_id", "")).strip()
    if not task_id:
        raise ValueError("spec must contain a non-empty task_id")
    text_instruction = str(spec.get("prompt_for_gen", "")).strip()
    if not text_instruction:
        raise ValueError(
            "spec must contain a non-empty prompt_for_gen (the text instruction)"
        )
    template = template_path.read_text(encoding="utf-8")
    replacements = {
        "{{TASK_ID}}": task_id,
        "{{TASK_LEVEL}}": level,
        "{{MIN_QAS}}": str(QA_LIMITS[level][0]),
        "{{MAX_QAS}}": str(QA_LIMITS[level][1]),
        "{{DIMENSIONS}}": ", ".join(DIMENSIONS),
        "{{IMPORTANCE_LEVELS}}": ", ".join(IMPORTANCE_LEVELS),
        "{{TEXT_INSTRUCTION}}": text_instruction,
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    unresolved = re.findall(r"\{\{[A-Z0-9_]+\}\}", template)
    if unresolved:
        raise ValueError(f"unresolved prompt placeholders {unresolved}")
    return template


def _parse_response(content: str) -> dict[str, Any]:
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM response does not contain a JSON object") from None
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object")
    return payload


def validate_qa_payload(
    payload: dict[str, Any], spec: dict[str, Any], level: str
) -> list[QAItem]:
    """Validate the model response and assign deterministic QA identifiers."""
    response_task_id = payload.get("task_id")
    if response_task_id != spec["task_id"]:
        raise ValueError(
            f"response task_id {response_task_id!r} does not match {spec['task_id']!r}"
        )
    response_level = str(payload.get("task_level", "")).upper()
    if response_level != level:
        raise ValueError(
            f"response task_level {response_level!r} does not match {level!r}"
        )
    raw_items = payload.get("qa_items")
    if not isinstance(raw_items, list):
        raise ValueError("LLM response must contain a qa_items list")

    min_qas, max_qas = QA_LIMITS[level]
    if not min_qas <= len(raw_items) <= max_qas:
        warnings.warn(
            f"{level} QA count {len(raw_items)} is outside the recommended "
            f"range {min_qas}-{max_qas}; the rules treat this as a guideline"
        )

    items: list[QAItem] = []
    normalized_questions: set[str] = set()
    for index, raw in enumerate(raw_items, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"QA item {index} must be an object")
        dimension_raw = str(raw.get("dimension", "")).strip()
        dimension = next(
            (d for d in DIMENSIONS if d.lower() == dimension_raw.lower()), ""
        )
        importance = str(raw.get("importance", "")).strip().lower()
        question = str(raw.get("question", "")).strip()
        if dimension not in DIMENSIONS:
            raise ValueError(f"QA item {index} has invalid dimension {dimension_raw!r}")
        if importance not in IMPORTANCE_LEVELS:
            raise ValueError(f"QA item {index} has invalid importance {importance!r}")
        if not question:
            raise ValueError(f"QA item {index} has an empty question")
        normalized = " ".join(question.lower().split())
        if normalized in normalized_questions:
            raise ValueError(f"QA item {index} duplicates an earlier question")
        normalized_questions.add(normalized)
        items.append(
            QAItem(
                id="",  # assigned after grouping by dimension
                dimension=dimension,
                importance=importance,
                question=question,
            )
        )

    missing_dimensions = sorted(set(DIMENSIONS) - {item.dimension for item in items})
    if missing_dimensions:
        raise ValueError(f"QA checklist misses dimensions {missing_dimensions}")
    if not any(item.importance == "critical" for item in items):
        raise ValueError("QA checklist must contain at least one critical item")

    # Group items by the canonical dimension order so the checklist reads in a
    # stable, dimension-grouped sequence, then assign sequential qa ids.
    dimension_order = {d: i for i, d in enumerate(DIMENSIONS)}
    items.sort(key=lambda item: dimension_order.get(item.dimension, len(DIMENSIONS)))
    for index, item in enumerate(items, start=1):
        item.id = f"q{index:02d}"
    return items


def load_qas(path: pathlib.Path) -> list[QAItem]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_items = payload.get("items", payload if isinstance(payload, list) else [])
    return [
        QAItem(
            id=item.get("qa_id", item.get("id", f"q{index + 1:02d}")),
            dimension=item["dimension"],
            importance=item["importance"],
            question=item["question"],
        )
        for index, item in enumerate(raw_items)
    ]
