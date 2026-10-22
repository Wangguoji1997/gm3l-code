#!/usr/bin/env python3
"""Prepare new, label-isolated train/test inputs from explicit user NPZ files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gm3l_real.data_io import load_dataset
from gm3l_real.preprocessing import prepare_splits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path, help="NPZ with X, A, ids/chat_id, y")
    source.add_argument("--embeddings", type=Path, help="NPZ with X, ids/chat_id, and optionally y")
    parser.add_argument("--adjacency", type=Path, help="NPZ with A and its own IDs; required with --embeddings")
    parser.add_argument("--evaluation", type=Path, help="Separate NPZ with y and IDs, for --embeddings")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a JSON object")
        dataset = load_dataset(args.dataset, embeddings_path=args.embeddings,
                               adjacency_path=args.adjacency, evaluation_path=args.evaluation)
        splits = prepare_splits(dataset, args.output, config)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Preparation failed: {exc}\n")
    print(f"Prepared {len(splits)} new splits in {args.output}; labels are outside input/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
