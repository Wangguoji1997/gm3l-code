#!/usr/bin/env python3
"""Execute a configured external method against an explicit label-free CSV directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gm3l_real.external import ExternalAdapterError, load_model_input_directory, run_external


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="User-created external config JSON; never an old experiment config")
    parser.add_argument("--input", type=Path, required=True, help="Directory containing only allowed model CSVs")
    parser.add_argument("--output", type=Path, required=True, help="New or empty output directory")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    allowed = {"command", "timeout", "cwd", "transductive"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError(f"External config accepts only {sorted(allowed)}")
    transductive = config.get("transductive", False)
    if not isinstance(transductive, bool):
        raise ValueError("transductive must be a JSON boolean")
    arrays = load_model_input_directory(args.input, transductive=transductive)
    try:
        result = run_external(config.get("command"), arrays, args.output, args.seed,
                              timeout=config.get("timeout", 3600), cwd=config.get("cwd"),
                              transductive=transductive)
    except ExternalAdapterError as exc:
        print(json.dumps({"status": exc.status, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": result["diagnostics"].get("status"), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
