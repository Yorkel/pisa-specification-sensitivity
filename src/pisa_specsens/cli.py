"""Command line entry point for running the specification grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .aggregate import summarise
from .config import RANDOM_SEED, build_grid
from .data import DataNotFoundError, load_pisa
from .grid import run_grid
from .preprocessing import prepare


class OutputDirectoryNotEmpty(RuntimeError):
    """Raised when the destination already holds results."""


def require_empty_output(path: Path) -> Path:
    """Refuse to write into a destination that already holds files.

    A results directory names a version. Overwriting one silently would make a
    published figure impossible to trace back to the run that produced it.
    """
    if path.exists() and any(path.iterdir()):
        raise OutputDirectoryNotEmpty(
            f"Output directory {path} is not empty. "
            "Choose a new versioned directory rather than overwriting results."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _progress(done: int, total: int, spec) -> None:
    print(f"[{done:>3}/{total}] {spec.cell_id}", file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pisa-specsens",
        description="Run the PISA specification sensitivity grid.",
    )
    parser.add_argument("--data", required=True, type=Path, help="Path to the UK PISA extract")
    parser.add_argument("--out", required=True, type=Path, help="Versioned output directory")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--limit", type=int, default=None, help="Run only the first N cells, for a smoke test"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        out = require_empty_output(args.out)
    except OutputDirectoryNotEmpty as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        frame = load_pisa(args.data)
    except DataNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    splits, pv_frame = prepare(frame, seed=args.seed)
    specs = build_grid()
    if args.limit is not None:
        specs = specs[: args.limit]

    print(
        f"rows {len(frame)}, features {len(splits.feature_names)}, cells {len(specs)}",
        file=sys.stderr,
    )
    results = run_grid(specs, splits, pv_frame, seed=args.seed, progress=_progress)

    results.to_csv(out / "grid_results.csv", index=False)
    results.to_json(out / "grid_results.json", orient="records", indent=2)
    summary = summarise(results)
    summary["run"] = {
        "n_rows": int(len(frame)),
        "n_features": len(splits.feature_names),
        "features": splits.feature_names,
        "seed": args.seed,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
