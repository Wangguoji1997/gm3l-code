#!/usr/bin/env python3
"""Copy existing embeddings or explicitly encode text with an OFFLINE local model.

The optional local encoder retains the source recipe: unmodified full text,
CLS pooling, float32 L2-normalized vectors, no query prefix and no truncation.
It never downloads a model or calls a hosted service. Supply model files yourself.
Labels are neither read by the encoder nor included in its output.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gm3l_real.data_io import load_embeddings, save_archive, validate_ids


def read_text_csv(path, id_column, text_column):
    ids, texts = [], []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if not {id_column, text_column}.issubset(rows.fieldnames or []):
            raise ValueError("Text CSV lacks the requested ID/text columns")
        for line, row in enumerate(rows, start=2):
            text = row[text_column]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty/non-string text at CSV line {line}")
            ids.append(row[id_column])
            texts.append(text)
    return validate_ids(np.asarray(ids, dtype=str)), texts


def encode_local(texts, model_dir, *, batch_size=4, max_tokens=8192, device="cpu"):
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise ValueError("--model-dir must be an existing local model directory, not a repository ID")
    if batch_size < 1 or max_tokens < 1:
        raise ValueError("batch_size and max_tokens must be positive")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Optional dependencies are imported only for an explicit encode command.
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    tokens = tokenizer(texts, add_special_tokens=True, padding=False, truncation=False)
    lengths = np.asarray([len(item) for item in tokens["input_ids"]])
    if np.any(lengths > max_tokens):
        raise ValueError("One or more texts exceed max_tokens; stopped without truncation or model inference")
    model = AutoModel.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False,
                                     torch_dtype=torch.float32).eval().to(device)
    position_limit = getattr(model.config, "max_position_embeddings", None)
    if isinstance(position_limit, int) and np.any(lengths > position_limit):
        raise ValueError("A text exceeds the local model's positional limit; no truncation performed")
    vectors = None
    order = np.argsort(lengths, kind="stable")
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        items = [{key: values[int(i)] for key, values in tokens.items()} for i in indices]
        batch = tokenizer.pad(items, padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state[:, 0]
            block = torch.nn.functional.normalize(hidden.float(), p=2, dim=-1).cpu().numpy()
        if vectors is None:
            vectors = np.empty((len(texts), block.shape[1]), dtype=np.float32)
        vectors[indices] = block
    if vectors is None or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=2e-5, rtol=0):
        raise ValueError("Invalid local embedding vectors")
    return vectors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    existing = sub.add_parser("import", help="Read existing vectors, strip text and evaluation labels")
    existing.add_argument("--embeddings", type=Path, required=True)
    existing.add_argument("--output", type=Path, required=True)
    local = sub.add_parser("encode-local", help="Explicitly run a user-supplied local model; no network")
    local.add_argument("--texts-csv", type=Path, required=True)
    local.add_argument("--id-column", default="id")
    local.add_argument("--text-column", default="text")
    local.add_argument("--model-dir", type=Path, required=True)
    local.add_argument("--output", type=Path, required=True)
    local.add_argument("--batch-size", type=int, default=4)
    local.add_argument("--max-tokens", type=int, default=8192)
    local.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError("Output exists; choose a new output filename")
        if args.command == "import":
            X, ids = load_embeddings(args.embeddings)
            metadata = {"mode": "user_supplied_embeddings", "labels_included": False}
        else:
            ids, texts = read_text_csv(args.texts_csv, args.id_column, args.text_column)
            X = encode_local(texts, args.model_dir, batch_size=args.batch_size,
                             max_tokens=args.max_tokens, device=args.device)
            metadata = {"mode": "optional_local_generation", "pooling": "CLS", "l2_normalized": True,
                        "text_preprocessing": "none", "truncation": False, "max_tokens": args.max_tokens,
                        "dtype": "float32", "labels_included": False,
                        "note": "User-supplied model; does not establish reproduction of a historical model snapshot"}
        save_archive(args.output, X=X, ids=ids, metadata_json=np.asarray(json.dumps(metadata)))
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        parser.exit(2, f"Feature preparation failed: {exc}\n")
    print(f"Saved X and IDs only to {args.output}; supply evaluation labels separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
