"""Batch runner for headless Fiji TrackMate automation.

This script scans experiment directories under an input root and executes
`automate_trackmate.py` in frame windows.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

FRAME_PATTERN = re.compile(r"^out_(\d+)\.tif$", re.IGNORECASE)


@dataclass(frozen=True)
class TrackmateParams:
    radius: float
    quality_thresh: float
    allow_track_splitting: int
    link_dist: float
    gap_dist: float
    max_frame_gap: int


@dataclass(frozen=True)
class BatchConfig:
    fiji_binary: Path
    script_path: Path
    input_root: Path
    sequence_subdir: str
    output_subdir: str
    step: int
    xmx: str
    params: TrackmateParams
    dry_run: bool
    strict: bool


def parse_args() -> BatchConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fiji", required=True, type=Path, help="Path to Fiji executable")
    parser.add_argument(
        "--script",
        type=Path,
        default=Path(__file__).with_name("automate_trackmate.py"),
        help="Path to automate_trackmate.py",
    )
    parser.add_argument("--input-root", required=True, type=Path, help="Parent folder containing experiment directories")
    parser.add_argument("--sequence-subdir", default="tifs", help="Subdirectory containing out_*.tif files")
    parser.add_argument("--output-subdir", default="results", help="Subdirectory used for TrackMate CSV output")
    parser.add_argument("--step", type=int, default=2000, help="Frames per batch")
    parser.add_argument("--xmx", default="18g", help="Java heap size passed as -Xmx")

    parser.add_argument("--radius", type=float, default=6.0)
    parser.add_argument("--quality-thresh", type=float, default=271.79)
    parser.add_argument("--allow-track-splitting", type=int, choices=(0, 1), default=1)
    parser.add_argument("--link-dist", type=float, default=45.0)
    parser.add_argument("--gap-dist", type=float, default=45.0)
    parser.add_argument("--max-frame-gap", type=int, default=2)

    parser.add_argument("--dry-run", action="store_true", help="Print commands without running Fiji")
    parser.add_argument("--strict", action="store_true", help="Exit on first experiment error")

    args = parser.parse_args()

    if args.step <= 0:
        parser.error("--step must be greater than 0")

    params = TrackmateParams(
        radius=args.radius,
        quality_thresh=args.quality_thresh,
        allow_track_splitting=args.allow_track_splitting,
        link_dist=args.link_dist,
        gap_dist=args.gap_dist,
        max_frame_gap=args.max_frame_gap,
    )

    return BatchConfig(
        fiji_binary=args.fiji.expanduser().resolve(),
        script_path=args.script.expanduser().resolve(),
        input_root=args.input_root.expanduser().resolve(),
        sequence_subdir=args.sequence_subdir,
        output_subdir=args.output_subdir,
        step=args.step,
        xmx=args.xmx,
        params=params,
        dry_run=args.dry_run,
        strict=args.strict,
    )


def discover_experiment_dirs(input_root: Path) -> list[Path]:
    return sorted(path for path in input_root.iterdir() if path.is_dir() and not path.name.startswith("._"))


def frame_tokens(sequence_dir: Path) -> list[int]:
    tokens: list[int] = []
    for entry in sequence_dir.iterdir():
        if not entry.is_file():
            continue
        match = FRAME_PATTERN.match(entry.name)
        if match:
            tokens.append(int(match.group(1)))
    return sorted(tokens)


def batch_ranges(tokens: list[int], step: int) -> Iterable[tuple[int, int]]:
    if not tokens:
        return

    idx = 0
    while idx < len(tokens):
        start = tokens[idx]
        end = tokens[min(idx + step - 1, len(tokens) - 1)]
        yield start, end
        idx += step


def build_fiji_command(config: BatchConfig, sequence_dir: Path, output_dir: Path, start: int, end: int) -> list[str]:
    p = config.params
    run_args = (
        f"seqDir='{sequence_dir}',"
        f"outDir='{output_dir}',"
        f"start={start},"
        f"end={end},"
        f"RADIUS={p.radius},"
        f"QUALITY_THRESH={p.quality_thresh},"
        f"ALLOW_TRACK_SPLITTING={p.allow_track_splitting},"
        f"LINK_DIST={p.link_dist},"
        f"GAP_DIST={p.gap_dist},"
        f"MAX_FRAME_GAP={p.max_frame_gap}"
    )
    return [
        str(config.fiji_binary),
        f"-Xmx{config.xmx}",
        "--headless",
        "--run",
        str(config.script_path),
        run_args,
    ]


def run_experiment(config: BatchConfig, experiment_dir: Path) -> bool:
    sequence_dir = experiment_dir / config.sequence_subdir
    output_dir = experiment_dir / config.output_subdir

    if not sequence_dir.is_dir():
        print(f"[SKIP] {experiment_dir}: missing sequence directory {sequence_dir}")
        return True

    tokens = frame_tokens(sequence_dir)
    if not tokens:
        print(f"[SKIP] {experiment_dir}: no out_<n>.tif files in {sequence_dir}")
        return True

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] {experiment_dir.name}: {len(tokens)} frames discovered")

    ok = True
    for start, end in batch_ranges(tokens, config.step):
        cmd = build_fiji_command(config, sequence_dir, output_dir, start, end)
        print(f"[RUN] {experiment_dir.name}: frames {start}..{end}")
        print("      ", shlex.join(cmd))
        if config.dry_run:
            continue

        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            ok = False
            print(f"[ERROR] {experiment_dir.name}: command failed with code {completed.returncode}")
            if config.strict:
                return False

    return ok


def validate_paths(config: BatchConfig) -> None:
    if not config.fiji_binary.exists():
        raise FileNotFoundError(f"Fiji binary not found: {config.fiji_binary}")
    if not config.script_path.exists():
        raise FileNotFoundError(f"TrackMate script not found: {config.script_path}")
    if not config.input_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {config.input_root}")


def main() -> int:
    config = parse_args()

    try:
        validate_paths(config)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"[FATAL] {exc}")
        return 2

    experiments = discover_experiment_dirs(config.input_root)
    if not experiments:
        print(f"[WARN] No experiment directories under {config.input_root}")
        return 0

    failures = 0
    for exp_dir in experiments:
        success = run_experiment(config, exp_dir)
        if not success:
            failures += 1
            if config.strict:
                break

    if failures:
        print(f"[DONE] Completed with {failures} failed experiment(s).")
        return 1

    print("[DONE] Completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
