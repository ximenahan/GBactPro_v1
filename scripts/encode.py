#!/usr/bin/env python3
"""Shared DNA one-hot encoding and padding for GBactpro models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from tensorflow.keras.utils import pad_sequences
except ImportError:  # pragma: no cover
    from keras.utils import pad_sequences  # type: ignore

_OH_BYTE = np.zeros((256, 4), dtype=np.float32)
for _b, _v in (
    ("A", (1, 0, 0, 0)),
    ("T", (0, 1, 0, 0)),
    ("C", (0, 0, 1, 0)),
    ("G", (0, 0, 0, 1)),
):
    _OH_BYTE[ord(_b)] = _v
    _OH_BYTE[ord(_b.lower())] = _v


def encode_sequence(seq: str) -> np.ndarray:
    """Convert a DNA string to one-hot array of shape (L, 4). Non-ACGT -> zeros."""
    if not seq:
        return np.zeros((0, 4), dtype=np.float32)
    b = np.frombuffer(seq.encode("latin-1"), dtype=np.uint8)
    return _OH_BYTE[b].copy()


def load_metadata(model_dir: Path) -> dict:
    meta_path = Path(model_dir) / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {model_dir}")
    with open(meta_path, encoding="utf-8") as fh:
        return json.load(fh)


def validate_length(seq: str, min_len: int, max_len: int) -> None:
    n = len(seq)
    if n > max_len:
        raise ValueError(
            f"Sequence length {n} exceeds model maxlen {max_len}. "
            f"Provide windows of {min_len}-{max_len} bp."
        )
    if n < min_len:
        raise ValueError(
            f"Sequence length {n} is shorter than expected minimum {min_len}. "
            f"Provide windows of {min_len}-{max_len} bp."
        )


def encode_and_pad(
    sequences: Sequence[str],
    maxlen: int,
    padding: str = "post",
    *,
    min_len=None,
    max_len=None,
) -> np.ndarray:
    """One-hot encode and pad a batch of DNA strings to shape (N, maxlen, 4)."""
    lo = min_len if min_len is not None else 1
    hi = max_len if max_len is not None else maxlen
    encoded: List[np.ndarray] = []
    for seq in sequences:
        s = seq.upper().replace("U", "T")
        if "N" in s:
            raise ValueError("Sequences containing N are not supported.")
        validate_length(s, lo, hi)
        encoded.append(encode_sequence(s))
    if not encoded:
        return np.zeros((0, maxlen, 4), dtype=np.float32)
    return np.asarray(
        pad_sequences(encoded, padding=padding, maxlen=maxlen),
        dtype=np.float32,
    )


def read_fasta(path: Path) -> List[Tuple[str, str]]:
    """Return list of (id, sequence) from a FASTA file (Biopython optional)."""
    path = Path(path)
    try:
        from Bio import SeqIO

        return [(rec.id, str(rec.seq).upper()) for rec in SeqIO.parse(str(path), "fasta")]
    except ImportError:
        records: List[Tuple[str, str]] = []
        seq_id = None
        chunks: List[str] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if seq_id is not None:
                        records.append((seq_id, "".join(chunks).upper()))
                    seq_id = line[1:].split()[0]
                    chunks = []
                else:
                    chunks.append(line)
            if seq_id is not None:
                records.append((seq_id, "".join(chunks).upper()))
        return records


def iter_batches(items: Sequence, batch_size: int) -> Iterable[Sequence]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
