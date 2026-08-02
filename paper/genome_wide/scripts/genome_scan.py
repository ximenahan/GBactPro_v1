#!/usr/bin/env python3
"""
Double-strand sliding-window genome scan with interval BED/TSV output.

- Forward (+): model input = genome[i : i+L] (0-based i; L = model input length).
- Reverse (−): model input = reverse_complement(genome[i : i+L]) (same genomic interval).
- Genomic interval for both strands (BED): start = i (0-based), end = i + L (exclusive).

Default: only windows with score >= threshold are written (+ BED + full TSV).

--all-windows: write every non-N window for both strands (no threshold filter).
--region-start / --region-end: optional 0-based half-open genomic slice (quick demos).

Inference: each batch runs one model.predict on [forward_batch ; reverse_batch] for speed.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences

_RC = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s: str) -> str:
    return s.translate(_RC)[::-1]


def sequence_to_onehot(sequence: str):
    enc = {"A": [1, 0, 0, 0], "T": [0, 1, 0, 0], "C": [0, 0, 1, 0], "G": [0, 0, 0, 1]}
    return [enc.get(n, [0, 0, 0, 0]) for n in sequence.upper()]


def encode_sequences_batch(sequences, seq_length: int, padding: str = "post"):
    encoded = []
    for seq in sequences:
        if "N" in seq.upper():
            continue
        encoded.append(sequence_to_onehot(seq))
    if not encoded:
        return np.array([])
    return np.array(pad_sequences(encoded, padding=padding, maxlen=seq_length), dtype=np.float32)


def prob_positive(preds, row_idx: int) -> float:
    row = preds[row_idx]
    return float(row[0]) if row.shape[-1] == 1 else float(row[1])


def bed_score_from_prob(p: float) -> int:
    """BED score column: 0–1000."""
    return int(max(0, min(1000, round(p * 1000))))


def run_scan(
    genome_file: str,
    model_path: str,
    threshold: float,
    chrom_name: str | None,
    out_bed: Path,
    out_tsv: Path,
    *,
    emit_all_windows: bool = False,
    region_start: int | None = None,
    region_end: int | None = None,
    padding: str = "post",
) -> tuple[int, int]:
    print("=" * 60)
    if emit_all_windows:
        print("Double-strand genome scan — all windows (BED + TSV)")
    else:
        print("Double-strand genome scan — thresholded windows (BED + TSV)")
        print(f"  threshold = {threshold}")
    print("=" * 60)

    model = load_model(model_path)
    seq_length = int(model.input_shape[1])
    print(f"  model window length = {seq_length}  padding = {padding}")

    genome_record = next(SeqIO.parse(genome_file, "fasta"))
    genome_sequence = str(genome_record.seq).upper()
    genome_id = chrom_name or genome_record.id
    genome_length = len(genome_sequence)

    # Optional genomic region [region_start, region_end) in 0-based coords
    scan_start = 0 if region_start is None else max(0, int(region_start))
    scan_end = genome_length if region_end is None else min(genome_length, int(region_end))
    if scan_end - scan_start < seq_length:
        raise ValueError(
            f"Region [{scan_start}, {scan_end}) shorter than window size {seq_length}"
        )
    if scan_start > 0 or scan_end < genome_length:
        print(f"  region (0-based half-open): [{scan_start}, {scan_end})")

    n_starts = (scan_end - seq_length + 1) - scan_start
    if n_starts <= 0:
        raise ValueError("Genome/region shorter than window size")

    batch_size = max(64, int(os.environ.get("SCAN_INFER_BATCH_SIZE", "2048")))

    rows: list[dict] = []
    n_fwd = 0
    n_rev = 0
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    full_row_fields = [
        "chrom",
        "start_0based",
        "end_0based_exclusive",
        "start_1based",
        "end_1based_inclusive",
        "strand",
        "score",
        "bed_score",
        "window_index_0based",
        "sequence_genomic_plus",
        "sequence_model_input",
    ]

    emit_fh = None
    bed_fh = None
    emit_writer: csv.DictWriter | None = None
    if emit_all_windows:
        out_bed.parent.mkdir(parents=True, exist_ok=True)
        emit_fh = open(out_tsv, "w", newline="", encoding="utf-8")
        bed_fh = open(out_bed, "w", encoding="utf-8")
        emit_writer = csv.DictWriter(
            emit_fh,
            fieldnames=full_row_fields,
            delimiter="\t",
            extrasaction="ignore",
        )
        emit_writer.writeheader()

    try:
        for bi in range(0, n_starts, batch_size):
            be = min(bi + batch_size, n_starts)
            fwd_seqs: list[str] = []
            rev_seqs: list[str] = []
            starts: list[int] = []

            for off in range(bi, be):
                i = scan_start + off
                w = genome_sequence[i : i + seq_length]
                if "N" in w:
                    continue
                fwd_seqs.append(w)
                rev_seqs.append(revcomp(w))
                starts.append(i)

            if not fwd_seqs:
                continue

            xf = encode_sequences_batch(fwd_seqs, seq_length, padding=padding)
            xr = encode_sequences_batch(rev_seqs, seq_length, padding=padding)
            if xf.size == 0 or xr.size == 0:
                continue
            x = np.concatenate([xf, xr], axis=0)
            preds = model.predict(x, verbose=0)
            nwin = len(starts)
            p_plus = preds[:nwin]
            p_minus = preds[nwin:]

            for j, i0 in enumerate(starts):
                s_plus = prob_positive(p_plus, j)
                s_minus = prob_positive(p_minus, j)

                start_bed = i0
                end_bed = i0 + seq_length
                start_1b = i0 + 1
                end_1b = i0 + seq_length

                seq_plus = fwd_seqs[j]
                seq_minus_input = rev_seqs[j]

                if emit_all_windows:
                    assert emit_writer is not None and bed_fh is not None
                    bp = bed_score_from_prob(s_plus)
                    bm = bed_score_from_prob(s_minus)
                    n_fwd += 1
                    emit_writer.writerow(
                        {
                            "chrom": genome_id,
                            "start_0based": start_bed,
                            "end_0based_exclusive": end_bed,
                            "start_1based": start_1b,
                            "end_1based_inclusive": end_1b,
                            "strand": "+",
                            "score": s_plus,
                            "bed_score": bp,
                            "window_index_0based": i0,
                            "sequence_genomic_plus": seq_plus,
                            "sequence_model_input": seq_plus,
                        }
                    )
                    bed_fh.write(
                        f"{genome_id}\t{start_bed}\t{end_bed}\twin{i0}_+\t{bp}\t+\n"
                    )
                    n_rev += 1
                    emit_writer.writerow(
                        {
                            "chrom": genome_id,
                            "start_0based": start_bed,
                            "end_0based_exclusive": end_bed,
                            "start_1based": start_1b,
                            "end_1based_inclusive": end_1b,
                            "strand": "-",
                            "score": s_minus,
                            "bed_score": bm,
                            "window_index_0based": i0,
                            "sequence_genomic_plus": seq_plus,
                            "sequence_model_input": seq_minus_input,
                        }
                    )
                    bed_fh.write(
                        f"{genome_id}\t{start_bed}\t{end_bed}\twin{i0}_-\t{bm}\t-\n"
                    )
                else:
                    if s_plus >= threshold:
                        n_fwd += 1
                        rows.append(
                            {
                                "chrom": genome_id,
                                "start_0based": start_bed,
                                "end_0based_exclusive": end_bed,
                                "start_1based": start_1b,
                                "end_1based_inclusive": end_1b,
                                "strand": "+",
                                "score": s_plus,
                                "bed_score": bed_score_from_prob(s_plus),
                                "window_index_0based": i0,
                                "sequence_genomic_plus": seq_plus,
                                "sequence_model_input": seq_plus,
                            }
                        )
                    if s_minus >= threshold:
                        n_rev += 1
                        rows.append(
                            {
                                "chrom": genome_id,
                                "start_0based": start_bed,
                                "end_0based_exclusive": end_bed,
                                "start_1based": start_1b,
                                "end_1based_inclusive": end_1b,
                                "strand": "-",
                                "score": s_minus,
                                "bed_score": bed_score_from_prob(s_minus),
                                "window_index_0based": i0,
                                "sequence_genomic_plus": seq_plus,
                                "sequence_model_input": seq_minus_input,
                            }
                        )
    finally:
        if emit_fh is not None:
            emit_fh.close()
        if bed_fh is not None:
            bed_fh.close()

    if emit_all_windows:
        total_rows = n_fwd + n_rev
        print(f"Rows (+): {n_fwd}  Rows (−): {n_rev}  Total rows: {total_rows}")
        print(f"Wrote BED: {out_bed}")
        print(f"Wrote TSV: {out_tsv}")
        return n_fwd, n_rev

    empty_cols = [
        "chrom",
        "start_0based",
        "end_0based_exclusive",
        "start_1based",
        "end_1based_inclusive",
        "strand",
        "score",
        "bed_score",
        "window_index_0based",
        "sequence_genomic_plus",
        "sequence_model_input",
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=empty_cols)
    df.to_csv(out_tsv, sep="\t", index=False)

    if out_bed is None:
        raise ValueError("out_bed required when not using --all-windows")
    out_bed.parent.mkdir(parents=True, exist_ok=True)
    with open(out_bed, "w") as fb:
        for _, r in df.iterrows():
            name = f"win{r['window_index_0based']}_{r['strand']}"
            fb.write(
                f"{r['chrom']}\t{int(r['start_0based'])}\t{int(r['end_0based_exclusive'])}\t"
                f"{name}\t{int(r['bed_score'])}\t{r['strand']}\n"
            )
    print(f"Hits (+): {n_fwd}  Hits (−): {n_rev}  Total rows: {len(df)}")
    print(f"Wrote BED: {out_bed}")
    print(f"Wrote TSV: {out_tsv}")
    return n_fwd, n_rev


def main():
    repo_root = Path(__file__).resolve().parents[3]
    default_model = repo_root / "models" / "type1_pre_max29bp_random"
    out_dir = Path(__file__).resolve().parent.parent / "output" / "03_scan"

    ap = argparse.ArgumentParser(
        description="Double-strand sliding-window genome scan for GBactpro"
    )
    ap.add_argument("--genome", required=True)
    ap.add_argument("--model", default=str(default_model))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument(
        "--all-windows",
        action="store_true",
        help="Emit every non-N window on both strands (no threshold); BED + full TSV (streamed)",
    )
    ap.add_argument("--chrom", default=None, help="Override FASTA seq id for output")
    ap.add_argument("--out-dir", type=Path, default=out_dir)
    ap.add_argument(
        "--prefix",
        default="NC_000913.2_type1_doublestrand",
        help="Output file name prefix (no extension)",
    )
    ap.add_argument(
        "--region-start",
        type=int,
        default=None,
        help="Optional 0-based inclusive start of genomic region to scan",
    )
    ap.add_argument(
        "--region-end",
        type=int,
        default=None,
        help="Optional 0-based exclusive end of genomic region to scan",
    )
    ap.add_argument(
        "--padding",
        choices=("pre", "post"),
        default="post",
        help="Pad strategy when window length < model maxlen (default post; "
        "equal-length windows make this a no-op)",
    )
    args = ap.parse_args()

    # Do not use Path.with_suffix — prefixes like NC_000913.2_* contain dots.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.all_windows:
        stem = f"{args.prefix}_allwindows"
    else:
        thr_tag = f"{args.threshold:g}".replace(".", "p")
        stem = f"{args.prefix}_thr{thr_tag}"
    bed_path = args.out_dir / f"{stem}.bed"
    tsv_path = args.out_dir / f"{stem}.tsv"

    run_scan(
        args.genome,
        args.model,
        args.threshold,
        args.chrom,
        bed_path,
        tsv_path,
        emit_all_windows=args.all_windows,
        region_start=args.region_start,
        region_end=args.region_end,
        padding=args.padding,
    )


if __name__ == "__main__":
    main()
