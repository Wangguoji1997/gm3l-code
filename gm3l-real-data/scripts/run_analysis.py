#!/usr/bin/env python3
"""Fit selected methods on prepared splits, preserving unsuccessful runs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--methods", required=True, nargs="+")
    parser.add_argument("--splits", nargs="+", help="Optional split IDs, e.g. 0 1 or split_000 split_001")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    from threadpoolctl import threadpool_limits
    from gm3l_real.runner import run_split
    if args.splits:
        paths = [args.prepared/(f"split_{int(s):03d}" if s.isdigit() else s) for s in args.splits]
    else:
        paths = sorted(args.prepared.glob("split_*/input/config.json"))
        paths = [path.parent.parent for path in paths]
    if not paths:
        parser.error("No prepared splits found")
    failed = False
    with threadpool_limits(limits=args.threads):
        for split in paths:
            reports = run_split(split/"input", args.runs/split.name, args.methods)
            for method, report in reports.items():
                print(f"{split.name}\t{method}\t{report['status']}", flush=True)
                failed |= report["status"] != "complete"
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
