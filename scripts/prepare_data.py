#!/usr/bin/env python3
"""Rebuild the benchmark media locally from the source videos.

The benchmark releases **links and annotations only** -- never the source videos
themselves, nor the clips and first frames derived from them.  Everything a
model actually consumes is regenerated here, on your machine, from the sources
you obtained under their own terms.

What this script needs from you: the source videos, downloaded yourself and named
by ``source_video_id`` (the bare three-digit source number):

    data/source_videos/<source_video_id>.<ext>      e.g. 001.mp4

``data/video_sources.csv`` maps each id to its URL / platform, and
``data/source_annotations/<source_video_id>.txt`` gives the clip spans
(``start_sec,end_sec,level``) **in the same order as that source's task_ids**
sorted by leading task number.  ``--check`` verifies that pairing before any
ffmpeg call.

Produces, keyed by the released ``task_id``:

    data/video_clips/<task_id>.mp4      the reference clip
    data/first_frames/<task_id>.jpg     its first frame -- the I0 model input

Only the first frame is required to run the benchmark; the clip itself is the
reference execution, which you need only if you want to re-derive annotations.

Source platforms differ in how you obtain a video, and NOT all of them are
downloadable:

  * bilibili / youtube -- download with any tool you are entitled to use
    (e.g. yt-dlp), in accordance with the platform's terms.
  * finebio / expvid   -- these come from third-party research datasets and must
    be fetched from the original release under its own licence.  The
    ``extra_id`` column in video_sources.csv is where the per-clip identifier
    for those sources belongs.

Usage:
    python scripts/prepare_data.py --check            # verify pairing + report what is missing
    python scripts/prepare_data.py --clips            # cut every clip
    python scripts/prepare_data.py --first-frames     # extract I0 from every clip
    python scripts/prepare_data.py --clips --first-frames --limit 5   # smoke test
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import shutil
import subprocess
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SOURCES_CSV = DATA_DIR / "video_sources.csv"
ANNOTATIONS = DATA_DIR / "source_annotations"
SPECS = DATA_DIR / "specs"
SOURCE_VIDEOS = DATA_DIR / "source_videos"
CLIPS = DATA_DIR / "video_clips"
FIRST_FRAMES = DATA_DIR / "first_frames"

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".flv", ".avi", ".m4v", ".ts")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def load_sources() -> dict[str, str]:
    """source_video_id -> source_url."""
    with open(SOURCES_CSV, newline="", encoding="utf-8") as fh:
        return {r["source_video_id"]: r["source_url"] for r in csv.DictReader(fh)}


def load_annotation(source_id: str) -> list[tuple[float, float, str]]:
    """One source's clip spans, in file order."""
    path = ANNOTATIONS / f"{source_id}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"no annotation file: {path}")
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    spans = []
    for lineno, line in enumerate(lines[1:], start=2):   # line 1 is the header
        parts = line.split(",")
        if len(parts) != 3:
            raise ValueError(f"{path}:{lineno}: expected start,end,level -- got {line!r}")
        start, end, level = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if level not in ("L1", "L2"):
            raise ValueError(f"{path}:{lineno}: level must be L1 or L2 -- got {level!r}")
        spans.append((float(start), float(end), level))
    return spans


def tasks_by_source() -> dict[str, list[tuple[str, str]]]:
    """source number -> [(task_id, level)] sorted by leading task number.

    task_id is ``<task_no>_L<level>_<source_no>_<discipline>``; the source number
    is the third field and the discipline may itself contain an underscore.
    """
    by_source: dict[str, list[tuple[str, str]]] = {}
    for path in sorted(SPECS.glob("*_spec.json")):
        spec = json.loads(path.read_text())
        task_id = spec["task_id"]
        parts = task_id.split("_")
        source_no, level = parts[2], parts[1]
        by_source.setdefault(source_no, []).append((task_id, level))
    return {k: sorted(v, key=lambda x: int(x[0].split("_")[0])) for k, v in by_source.items()}


def find_source_video(source_id: str) -> pathlib.Path | None:
    for ext in VIDEO_EXTENSIONS:
        candidate = SOURCE_VIDEOS / f"{source_id}{ext}"
        if candidate.is_file():
            return candidate
    return None


def plan() -> tuple[list[tuple[str, pathlib.Path, float, float]],
                    list[tuple[str, str]], list[str]]:
    """Pair every task with its source video and span.

    Returns (jobs, pairing_problems, sources_without_video) where a job is
    (task_id, source_video_path, start_sec, end_sec) and a pairing problem is
    (source_id, message).
    """
    sources = load_sources()
    by_source = tasks_by_source()
    id_by_no = {sid.split("_")[0]: sid for sid in sources}
    jobs: list[tuple[str, pathlib.Path, float, float]] = []
    problems: list[tuple[str, str]] = []
    no_video: list[str] = []

    for no, pairs in sorted(by_source.items()):
        source_id = id_by_no.get(no)
        if source_id is None:
            problems.append((no, f"no row in {SOURCES_CSV.name}"))
            continue
        try:
            spans = load_annotation(source_id)
        except (FileNotFoundError, ValueError) as exc:
            problems.append((source_id, str(exc)))
            continue
        if len(spans) != len(pairs):
            problems.append((source_id,
                             f"{len(spans)} annotation rows for {len(pairs)} tasks"))
            continue

        video = find_source_video(source_id)
        if video is None:
            no_video.append(source_id)
            continue

        for (task_id, level), (start, end, span_level) in zip(pairs, spans):
            if level != span_level:
                problems.append((source_id,
                                 f"{task_id} is {level} but its span is {span_level}"))
                continue
            if end <= start:
                problems.append((source_id, f"{task_id} has an empty span {start}->{end}"))
                continue
            jobs.append((task_id, video, start, end))

    # A source that yields no tasks but has no local video is worth flagging too.
    for source_id in sorted(sources):
        if source_id.split("_")[0] not in by_source and find_source_video(source_id) is None:
            no_video.append(source_id)
    return jobs, problems, no_video


# ---------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------
def require_ffmpeg() -> None:
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        raise SystemExit(f"{', '.join(missing)} not found on PATH; install ffmpeg first")


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def cut_clip(video: pathlib.Path, start: float, end: float, out: pathlib.Path,
             overwrite: bool) -> bool:
    """Cut [start, end) to ``out``.  Returns False when skipped."""
    if out.exists() and not overwrite:
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    # Re-encode rather than stream-copy: the span boundaries are what the
    # annotations specify, and a copy can only cut on keyframes.
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y" if overwrite else "-n",
         "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{end - start:.3f}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-an",
         "-pix_fmt", "yuv420p", str(out)])
    return True


def extract_first_frame(clip: pathlib.Path, out: pathlib.Path, overwrite: bool) -> bool:
    if out.exists() and not overwrite:
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y" if overwrite else "-n",
         "-i", str(clip), "-frames:v", "1", "-q:v", "2", str(out)])
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Rebuild clips and first frames from locally obtained source videos.")
    ap.add_argument("--check", action="store_true",
                    help="verify annotation/task pairing and report missing source videos")
    ap.add_argument("--clips", action="store_true",
                    help=f"cut reference clips into {CLIPS.relative_to(PROJECT_ROOT)}/")
    ap.add_argument("--first-frames", action="store_true",
                    help=f"extract I0 into {FIRST_FRAMES.relative_to(PROJECT_ROOT)}/")
    ap.add_argument("--overwrite", action="store_true", help="redo files that already exist")
    ap.add_argument("--limit", type=int, default=0, help="process at most N tasks (0 = all)")
    args = ap.parse_args(argv)

    if not (args.check or args.clips or args.first_frames):
        ap.print_help()
        return 1

    jobs, problems, no_video = plan()

    print(f"tasks ready to cut              : {len(jobs)}")
    print(f"source videos still missing     : {len(no_video)}")
    if problems:
        print(f"\n!! {len(problems)} pairing problem(s) -- fix these before cutting:")
        for source_id, msg in problems[:20]:
            print(f"   {source_id}: {msg}")
        if len(problems) > 20:
            print(f"   ... and {len(problems) - 20} more")

    if args.check:
        if no_video:
            print("\nMissing source videos -- see data/README.md for where each "
                  "platform's videos come from:")
            for source_id in no_video[:20]:
                print(f"   {source_id}   -> data/source_videos/{source_id}.<ext>")
            if len(no_video) > 20:
                print(f"   ... and {len(no_video) - 20} more")
        print("\nNote: finebio / expvid sources are not web links -- they must be "
              "obtained from the original dataset release under its own licence.")
        return 0 if not problems else 2

    if problems:
        return 2
    if not jobs:
        print("\nnothing to do: no source videos found under "
              f"{SOURCE_VIDEOS.relative_to(PROJECT_ROOT)}/")
        return 1

    require_ffmpeg()
    if args.limit:
        jobs = jobs[: args.limit]

    n_cut = n_frame = n_skip = 0
    for i, (task_id, video, start, end) in enumerate(jobs, start=1):
        clip = CLIPS / f"{task_id}.mp4"
        frame = FIRST_FRAMES / f"{task_id}.jpg"
        try:
            if args.clips:
                if cut_clip(video, start, end, clip, args.overwrite):
                    n_cut += 1
                    print(f"[{i}/{len(jobs)}] clip  {task_id}  {start:.3f}->{end:.3f}s")
                else:
                    n_skip += 1
            if args.first_frames:
                if not clip.exists():
                    print(f"[{i}/{len(jobs)}] SKIP {task_id}: no clip yet "
                          f"(run --clips first)")
                    continue
                if extract_first_frame(clip, frame, args.overwrite):
                    n_frame += 1
                    print(f"[{i}/{len(jobs)}] frame {task_id}")
                else:
                    n_skip += 1
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or b"").decode(errors="replace").strip().splitlines()
            print(f"[{i}/{len(jobs)}] ERROR {task_id}: {detail[-1] if detail else exc}")

    print(f"\ncut {n_cut} clip(s), extracted {n_frame} frame(s), skipped {n_skip} existing.")
    if n_cut or n_frame:
        print("Next: python scripts/specs_to_tasks.py   # -> data/tasks.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
