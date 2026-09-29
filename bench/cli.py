"""LabInstruct benchmark CLI.

    python -m bench.cli gen    --run myrun --models wan2.2,ltx2.3 --gpus 0,1,2,3 --out /path/to/outputs
    python -m bench.cli task   --model wan2.2 --task-id 001_L1_001_agronomy

Run ``python -m bench.cli <cmd> --help`` for per-command options.
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

from bench import PROJECT_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("labinstruct")


def _csv(v: str | None) -> list[str] | None:
    if not v:
        return None
    return [x.strip() for x in v.split(",") if x.strip()]


def _default_tasks() -> str:
    """The converted benchmark tasks, built by scripts/specs_to_tasks.py."""
    return str(PROJECT_ROOT / "data" / "tasks.jsonl")


def cmd_gen(args: argparse.Namespace) -> int:
    from bench.core import io
    from bench.core.dispatcher import Dispatcher

    tasks = pathlib.Path(args.tasks)
    if not tasks.exists():
        raise SystemExit(
            f"no task file: {tasks}\n"
            "build it first:  python scripts/specs_to_tasks.py"
        )
    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else None
    dispatcher = Dispatcher(
        models=_csv(args.models),
        task_ids=_csv(args.task_ids),
        tasks_source=pathlib.Path(args.tasks),
        run_id=args.run,
        gpus=gpus,
        continue_=args.continue_,
        root=PROJECT_ROOT,
        outputs_root=pathlib.Path(args.out) if args.out else PROJECT_ROOT / "data" / "outputs",
    )
    results = dispatcher.run()
    n_ok = n_err = 0
    for rec in io.iter_results(results):
        if rec.get("status") == "ok":
            n_ok += 1
        elif rec.get("status") == "error":
            n_err += 1
    print(f"\nDone: {n_ok} ok, {n_err} errors -> {results}")
    return 0 if n_err == 0 else 1


def cmd_task(args: argparse.Namespace) -> int:
    """Run a single task through one adapter (debugging aid)."""
    from bench.adapters import build_adapter
    from bench.registry import load_models, load_tasks

    tasks = load_tasks(pathlib.Path(args.tasks), root=PROJECT_ROOT)
    task = next((t for t in tasks if t.task_id == args.task_id), tasks[0])
    cfg = load_models()[args.model]
    adapter = build_adapter(args.model, cfg, PROJECT_ROOT)
    result = adapter.generate(task, pathlib.Path(args.out), gpu=args.gpu)
    print(result)
    return 0 if result.ok else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m bench.cli", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("gen", help="run the generation benchmark")
    p.add_argument("--run", default=None, help="run id (default: timestamped)")
    p.add_argument("--models", default=None, help="comma-separated model ids; default all")
    p.add_argument("--tasks", default=_default_tasks())
    p.add_argument("--task-ids", default=None, help="comma-separated task ids to restrict to")
    p.add_argument("--gpus", default=None, help="GPU pool (comma-separated indices, e.g. 0,1,2,3); single-GPU jobs run concurrently one per card, multi-GPU models (num_gpus>1) run alone")
    p.add_argument("--out", default=None, help="output root for videos and run metadata (default: <project>/data/outputs)")
    p.add_argument("--continue", dest="continue_", action="store_true", help="skip already-completed jobs")
    p.set_defaults(func=cmd_gen)

    p = sub.add_parser("task", help="run a single task through one adapter (debug)")
    p.add_argument("--model", required=True)
    p.add_argument("--task-id", default=None)
    p.add_argument("--tasks", default=_default_tasks())
    p.add_argument("--out", default=str(PROJECT_ROOT / "data" / "outputs" / "_debug"))
    p.add_argument("--gpu", type=int, default=0)
    p.set_defaults(func=cmd_task)
    return ap


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
