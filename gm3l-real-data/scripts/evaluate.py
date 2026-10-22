#!/usr/bin/env python3
"""Evaluate exported runs; this is the only analysis stage reading human labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gm3l_real.evaluation import evaluate_experiment


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", help="Requested IDs (0 1 ... or split_000 ...); missing runs remain in reports")
    args = parser.parse_args(argv)
    summary = evaluate_experiment(args.prepared, args.runs, args.output, args.methods, args.splits)
    print(json.dumps({"paired_n": summary["paired_n"], "requested_n": len(summary["requested_split_ids"]),
                      "methods": summary["methods"], "output": str(args.output)}, ensure_ascii=False, indent=2))
    return 0 if summary["paired_n"] == len(summary["requested_split_ids"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
