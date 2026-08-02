#!/usr/bin/env python3
"""
GBactpro fixed-length promoter prediction CLI.

Predict whether each DNA window is a bacterial promoter (binary classification).
Default model: type1_35s10_random (27–31 bp windows, post-padded to maxlen 31).
"""
from __future__ import annotations

import os

# Must be set before TensorFlow is imported (including via encode.py).
# Level 3 hides the noisy "cuInit: CUDA_ERROR_NO_DEVICE" line on CPU-only machines.
# Override with: GBACTPRO_TF_VERBOSE=1
if os.environ.get("GBACTPRO_TF_VERBOSE") != "1":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from encode import encode_and_pad, load_metadata, read_fasta  # noqa: E402


def _default_model() -> Path:
    return REPO_ROOT / "models" / "type1_35s10_random"


def load_keras_model(model_dir: Path):
    try:
        from tensorflow.keras.models import load_model
    except ImportError:  # pragma: no cover
        from keras.models import load_model  # type: ignore
    return load_model(str(model_dir))


def predict_sequences(
    records,
    model_dir: Path,
    threshold=None,
    batch_size: int = 256,
):
    meta = load_metadata(model_dir)
    maxlen = int(meta["maxlen"])
    padding = str(meta.get("padding", "post"))
    lo, hi = meta.get("expected_length_range", [1, maxlen])
    default_thr = float(meta.get("default_threshold", 0.5))
    thr = default_thr if threshold is None else float(threshold)

    model = load_keras_model(model_dir)
    ids = [r[0] for r in records]
    seqs = [r[1] for r in records]

    scores = np.zeros(len(seqs), dtype=np.float64)
    for start in range(0, len(seqs), batch_size):
        batch = seqs[start : start + batch_size]
        x = encode_and_pad(batch, maxlen=maxlen, padding=padding, min_len=lo, max_len=hi)
        preds = model.predict(x, verbose=0)
        scores[start : start + len(batch)] = np.asarray(preds).reshape(-1)

    rows = []
    for i, (sid, seq) in enumerate(zip(ids, seqs)):
        score = float(scores[i])
        label = "promoter" if score >= thr else "non-promoter"
        rows.append(
            {
                "id": sid,
                "sequence": seq,
                "length": len(seq),
                "score": "{:.6f}".format(score),
                "prediction": label,
                "threshold": "{:.4f}".format(thr),
            }
        )
    return rows, meta


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="GBactpro: predict promoter probability for fixed-length DNA windows",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("-i", "--input", required=True, help="Input FASTA of DNA windows")
    ap.add_argument("-o", "--output", required=True, help="Output TSV path")
    ap.add_argument(
        "--model",
        default=str(_default_model()),
        help="Path to SavedModel directory (with metadata.json)",
    )
    ap.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold (default: value in metadata.json, usually 0.5)",
    )
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args(argv)

    model_dir = Path(args.model).resolve()
    if not (model_dir / "saved_model.pb").exists():
        sys.exit("ERROR: SavedModel not found at {} (missing saved_model.pb)".format(model_dir))

    records = read_fasta(Path(args.input))
    if not records:
        sys.exit("ERROR: no sequences found in {}".format(args.input))

    try:
        rows, meta = predict_sequences(
            records, model_dir, threshold=args.threshold, batch_size=args.batch_size
        )
    except ValueError as exc:
        sys.exit("ERROR: {}".format(exc))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "sequence", "length", "score", "prediction", "threshold"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    n_pos = sum(1 for r in rows if r["prediction"] == "promoter")
    print("Model        : {} ({})".format(model_dir.name, meta.get("description", "")))
    print("Sequences    : {}".format(len(rows)))
    print("Promoter call: {} / {}".format(n_pos, len(rows)))
    print("Wrote        : {}".format(out_path))


if __name__ == "__main__":
    main()
