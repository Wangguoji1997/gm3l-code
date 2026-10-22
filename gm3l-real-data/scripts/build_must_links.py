#!/usr/bin/env python3
"""Import complete user-supplied must-links; no API clients or hidden LLM calls.

The pair route requires an explicit 0/1 (or Yes/No) judgement for every unordered
pair. An absent pair is unknown, not a negative edge. No transitive closure is
performed. The provenance/consent of annotations must be established by the user.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gm3l_real.data_io import archive_ids, load_adjacency, adjacency_from_pairs, save_archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--adjacency", type=Path, help="Existing complete A + IDs NPZ")
    source.add_argument("--pairs", type=Path, help="Exhaustive CSV pair labels")
    parser.add_argument("--ids-npz", type=Path, help="NPZ with ids/chat_id; required with --pairs")
    parser.add_argument("--left-column", default="id_i")
    parser.add_argument("--right-column", default="id_j")
    parser.add_argument("--label-column", default="must_link")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.adjacency is not None:
            if args.ids_npz is not None:
                raise ValueError("--ids-npz is only used with --pairs")
            A, ids = load_adjacency(args.adjacency)
            mode = "user_supplied_adjacency"
        else:
            if args.ids_npz is None:
                raise ValueError("--pairs requires --ids-npz to define the complete record universe")
            with np.load(args.ids_npz, allow_pickle=False) as archive:
                ids = archive_ids(archive)
            A = adjacency_from_pairs(args.pairs, ids, left_column=args.left_column,
                                     right_column=args.right_column, label_column=args.label_column)
            mode = "user_supplied_exhaustive_pair_annotations"
        metadata = {"mode": mode, "diagonal": 0, "symmetric": True, "transitive_closure": False,
                    "api_calls": 0, "annotation_source": "user_to_document"}
        save_archive(args.output, A=A, ids=ids, complete=np.asarray(True),
                     metadata_json=np.asarray(json.dumps(metadata)))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Must-link import failed: {exc}\n")
    print(f"Saved complete adjacency to {args.output}; no generation service was called.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
