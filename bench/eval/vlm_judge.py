"""Prompt building and response validation for VLM judging.

The judge prompt is rendered from ``data/rules/vlm_judge.txt``: every QA item
gets exactly one verdict ("yes" / "no" / "unjudgeable") plus a confidence score
and a short evidence-based rationale. The QA checklist embedded in the prompt
follows the item schema defined by ``data/rules/qa_generation.txt`` (qa_id,
dimension, importance, question; see bench.eval.qa.QAItem).

Sending the request is left to ``scripts/judge_videos_{gpt,gemini}.py``, which
attach media in whatever form their endpoint accepts.
"""
from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass
from typing import Any

from bench import PROJECT_ROOT
from bench.eval.qa import QAItem

JUDGE_TEMPLATE = PROJECT_ROOT / "data" / "rules" / "vlm_judge.txt"

VERDICTS = ("yes", "no", "unjudgeable")


@dataclass
class QAJudgement:
    """One QA item's verdict, as returned by the judge."""

    qa_id: str
    dimension: str
    importance: str
    question: str  # frozen question, echoed verbatim for downstream analysis
    verdict: str  # "yes" | "no" | "unjudgeable"
    confidence: float  # 0.0-1.0, confidence in the verdict
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "qa_id": self.qa_id,
            "dimension": self.dimension,
            "importance": self.importance,
            "question": self.question,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


def build_judge_prompt(
    spec: dict[str, Any],
    qas: list[QAItem],
    level: str,
    *,
    video_text: str,
    template_path: pathlib.Path = JUDGE_TEMPLATE,
) -> str:
    """Render ``data/rules/vlm_judge.txt`` for one spec + frozen QA checklist.

    ``video_text`` replaces the media placeholder with a marker describing how
    the frames are attached to the message; the bytes themselves are sent by
    the caller, which knows what its endpoint accepts.
    """
    task_id = spec.get("task_id")
    if not task_id:
        raise ValueError("spec must contain a non-empty task_id")
    if not level:
        raise ValueError("task level must be non-empty")
    # QA checklist in the output format of data/rules/qa_generation.txt.
    checklist = {
        "task_id": task_id,
        "task_level": level,
        "qa_items": [qa.to_dict() for qa in qas],
    }
    template = template_path.read_text(encoding="utf-8")
    replacements = {
        "{{TASK_ID}}": str(task_id),
        "{{TASK_LEVEL}}": level,
        "{{SPEC_JSON}}": json.dumps(spec, ensure_ascii=False, indent=2),
        "{{QA_CHECKLIST_JSON}}": json.dumps(checklist, ensure_ascii=False, indent=2),
        "{{GENERATED_VIDEO}}": video_text,
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    unresolved = re.findall(r"\{\{[A-Z0-9_]+\}\}", template)
    if unresolved:
        raise ValueError(f"unresolved judge prompt placeholders {unresolved}")
    return template


def judge_payload(
    results: dict[str, QAJudgement], task_id: str, task_level: str
) -> dict[str, Any]:
    """Serialize results in the judge template's output format (for disk).

    Each result additionally echoes the frozen ``question`` after
    ``importance`` so downstream analyses have the full item context.
    """
    return {
        "task_id": task_id,
        "task_level": task_level,
        "qa_results": [r.to_dict() for r in results.values()],
    }


def _validate_judge_payload(
    payload: dict[str, Any], spec: dict[str, Any], level: str, qas: list[QAItem]
) -> dict[str, QAJudgement]:
    """Validate the judge response against the frozen checklist, in order."""
    response_task_id = payload.get("task_id")
    if response_task_id != spec["task_id"]:
        raise ValueError(
            f"response task_id {response_task_id!r} does not match {spec['task_id']!r}"
        )
    response_level = str(payload.get("task_level", "")).upper()
    if response_level != level.upper():
        raise ValueError(
            f"response task_level {response_level!r} does not match {level!r}"
        )
    raw_results = payload.get("qa_results")
    if not isinstance(raw_results, list):
        raise ValueError("judge response must contain a qa_results list")
    if len(raw_results) != len(qas):
        raise ValueError(
            f"judge returned {len(raw_results)} results for {len(qas)} QA items"
        )

    results: dict[str, QAJudgement] = {}
    for qa, raw in zip(qas, raw_results):
        if not isinstance(raw, dict):
            raise ValueError(f"judge result for {qa.id} must be an object")
        if raw.get("qa_id") != qa.id:
            raise ValueError(f"expected result for {qa.id}, got {raw.get('qa_id')!r}")
        if str(raw.get("dimension", "")).strip().lower() != qa.dimension.lower():
            raise ValueError(
                f"judge changed dimension of {qa.id}: {raw.get('dimension')!r}"
            )
        if str(raw.get("importance", "")).strip().lower() != qa.importance.lower():
            raise ValueError(
                f"judge changed importance of {qa.id}: {raw.get('importance')!r}"
            )
        verdict = str(raw.get("verdict", "")).strip().lower()
        if verdict not in VERDICTS:
            raise ValueError(f"{qa.id} has invalid verdict {raw.get('verdict')!r}")
        confidence = raw.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError(f"{qa.id} has invalid confidence {confidence!r}")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"{qa.id} confidence {confidence} outside [0.0, 1.0]")
        rationale = str(raw.get("rationale", "")).strip()
        if not rationale:
            raise ValueError(f"{qa.id} has an empty rationale")
        results[qa.id] = QAJudgement(
            qa_id=qa.id,
            dimension=qa.dimension,
            importance=qa.importance,
            question=qa.question,
            verdict=verdict,
            confidence=float(confidence),
            rationale=rationale,
        )
    return results
