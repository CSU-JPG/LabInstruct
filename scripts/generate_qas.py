#!/usr/bin/env python3
"""Draft QA checklists from LabInstruct task specs with an LLM.

The model reads ``data/rules/qa_generation.txt`` plus one ``*_spec.json`` and
writes one ``*_qa.json`` into ``data/checklists``, named after the same task id.

Examples
    # single spec
    python scripts/generate_qas.py data/specs/001_L1_001_agronomy_spec.json

    # batch (all specs)
    python scripts/generate_qas.py data/specs/*_spec.json

    # force regenerate even when the output file already exists
    python scripts/generate_qas.py data/specs/*_spec.json --overwrite

    # preview the rendered prompt without calling the API
    python scripts/generate_qas.py data/specs/001_L1_001_agronomy_spec.json --print-prompt

Credentials come from the environment; none are stored in this file:
    LABINSTRUCT_LLM_API_KEY   (falls back to DEEPSEEK_API_KEY)
The endpoint and model default to DeepSeek's public API and can be overridden
with --base-url / --model or the matching environment variables.

Every checklist this drafts is a *draft*. The paper's protocol requires a human
expert to verify each item before it is used -- see the human_eval directory for
the rating interface used afterwards.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI  # noqa: E402

from bench.eval.qa import (  # noqa: E402
    _parse_response,
    build_qa_prompt,
    load_spec,
    validate_qa_payload,
)

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_TEMPLATE = PROJECT_ROOT / "data" / "rules" / "qa_generation.txt"


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "checklists"


def qa_output_path(spec_path: pathlib.Path) -> pathlib.Path:
    """Map a spec filename to its QA output filename.

    ``012_L1_012_materials_science_spec.json`` -> ``012_L1_012_materials_science_qa.json``
    Specs not named ``*_spec.json`` get ``_qa.json`` appended to the stem.
    """
    stem = spec_path.stem
    if stem.endswith("_spec"):
        stem = stem[: -len("_spec")] + "_qa"
    else:
        stem = stem + "_qa"
    return spec_path.with_name(stem + ".json")


def llm_generate_json(prompt: str, llm_cfg: dict[str, str]) -> str:
    """Call an OpenAI-compatible chat-completions endpoint to draft a checklist."""
    client = OpenAI(
        api_key=llm_cfg["api_key"],
        base_url=llm_cfg.get("base_url") or DEFAULT_BASE_URL,
    )
    response = client.chat.completions.create(
        model=llm_cfg["model"],
        messages=[
            {
                "role": "system",
                "content": (
                    "You design atomic, visually verifiable laboratory video "
                    "QA checklists."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        stream=False,
        temperature=0.2,
        reasoning_effort="high",
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "enabled"}},
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("model returned an empty completion")
    return content


def generate_qa(
    spec_path: pathlib.Path,
    *,
    template_path: pathlib.Path,
    output_dir: pathlib.Path,
    llm_cfg: dict[str, str],
    overwrite: bool = False,
) -> tuple[pathlib.Path, bool]:
    """Generate one QA file from one spec file.

    Returns ``(out_path, wrote)`` where ``wrote`` is False when the output
    already exists and ``overwrite`` is off (the file is then skipped).
    """
    spec, level = load_spec(spec_path)
    out_path = output_dir / qa_output_path(spec_path).name
    if out_path.exists() and not overwrite:
        return out_path, False

    prompt = build_qa_prompt(spec, level, template_path)
    payload = _parse_response(llm_generate_json(prompt, llm_cfg))
    items = validate_qa_payload(payload, spec, level)
    output = {
        "task_id": spec["task_id"],
        "task_level": level,
        "source_spec": spec_path.name,
        "items": [item.to_dict() for item in items],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return out_path, True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "specs",
        nargs="+",
        type=pathlib.Path,
        help="one or more *-spec.json files (shell glob OK)",
    )
    parser.add_argument(
        "--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--template", type=pathlib.Path, default=DEFAULT_TEMPLATE
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LABINSTRUCT_LLM_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LABINSTRUCT_LLM_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY", ""),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LABINSTRUCT_LLM_MODEL", DEFAULT_MODEL),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="regenerate QA files that already exist",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="render prompts without loading a model or calling an API",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.print_prompt and not args.api_key:
        print(
            "error: set LABINSTRUCT_LLM_API_KEY or DEEPSEEK_API_KEY "
            "(or pass --api-key)",
            file=sys.stderr,
        )
        return 2

    template_path = args.template.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    llm_cfg = {
        "base_url": args.base_url,
        "api_key": args.api_key,
        "model": args.model,
    }

    failures = 0
    for raw_path in args.specs:
        spec_path = raw_path.expanduser().resolve()
        if not spec_path.is_file():
            print(f"error: spec file not found: {spec_path}", file=sys.stderr)
            failures += 1
            continue

        if args.print_prompt:
            if len(args.specs) > 1:
                print(f"===== {spec_path.name} =====")
            spec, level = load_spec(spec_path)
            print(build_qa_prompt(spec, level, template_path))
            continue

        try:
            out_path, wrote = generate_qa(
                spec_path,
                template_path=template_path,
                output_dir=output_dir,
                llm_cfg=llm_cfg,
                overwrite=args.overwrite,
            )
            status = "wrote" if wrote else "skip existing"
            print(f"{status} {out_path}")
        except Exception as exc:  # keep batch generation going on one bad spec
            print(f"error {spec_path.name}: {exc}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
