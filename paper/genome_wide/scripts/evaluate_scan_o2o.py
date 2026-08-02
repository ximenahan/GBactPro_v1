#!/usr/bin/env python3
"""
Strand-aware PR curve with one-to-one matching between predicted windows and
reference promoter regions.

Constraint (greedy, score-descending order — same spirit as event-level eval):
  • Each predicted window interval matches at most one reference TSS / promoter.
  • Each reference promoter matches at most one predicted window.

A window is eligible to match a TSS only if it is fully contained in that TSS’s
(tolerance-expanded) promoter region and strands match — same geometry as the
dense window-level script. Among eligible unmatched TSS candidates, the
nearest TSS by genomic distance (window center vs TSS) wins.

Processing all windows in global descending score order makes the matching for
the top-k predictions identical to “re-run greedy on exactly those k windows”.
"""
from __future__ import annotations

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

UPSTREAM_MIN = 6
UPSTREAM_MAX = 39
WINDOW_LEN = 29

# Plot styling (PR curve)
CURVE_LINEWIDTH = 5.0
LEGEND_FONTSIZE = 14


def load_predictions(tsv_path: str):
    print(f"Loading predictions: {tsv_path} ...")
    t0 = time.time()
    df = pd.read_csv(
        tsv_path, sep="\t",
        usecols=["start_1based", "end_1based_inclusive", "strand", "score"],
        dtype={"start_1based": np.int64, "end_1based_inclusive": np.int64, "score": np.float64},
    )
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    starts = df["start_1based"].values
    ends = df["end_1based_inclusive"].values
    is_plus = (df["strand"].values == "+")
    scores = df["score"].values
    print(f"  + strand: {is_plus.sum():,}  - strand: {(~is_plus).sum():,}")
    print(f"  Score range: [{scores.min():.6f}, {scores.max():.6f}]")
    return starts, ends, is_plus, scores


def load_reference_tss(csv_path: str):
    print(f"Loading reference TSS: {csv_path} ...")
    df = pd.read_csv(csv_path)
    positions = df["Pos"].values.astype(np.int64)
    is_plus = (df["Strand"].values == "+")
    print(f"  Total: {len(df)}  (+ {is_plus.sum():,}, - {(~is_plus).sum():,})")
    return positions, is_plus


def _build_promoter_regions(ref_pos, ref_plus, tolerance: int):
    regions = {}
    for strand_val, mask in [("+", ref_plus), ("-", ~ref_plus)]:
        pos = ref_pos[mask]
        if strand_val == "+":
            rs = pos - UPSTREAM_MAX - tolerance
            re = pos - UPSTREAM_MIN + tolerance
        else:
            rs = pos + UPSTREAM_MIN - tolerance
            re = pos + UPSTREAM_MAX + tolerance
        order = np.argsort(rs)
        rs, re = rs[order], re[order]
        prefix_max_end = np.maximum.accumulate(re)
        regions[strand_val] = (rs, re, prefix_max_end)
    return regions


def _vectorized_is_tp(w_starts, w_ends, reg_starts, reg_ends, prefix_max_end):
    n = len(w_starts)
    if n == 0 or len(reg_starts) == 0:
        return np.zeros(n, dtype=np.bool_)
    idx = np.searchsorted(reg_starts, w_starts, side="right") - 1
    valid = idx >= 0
    result = np.zeros(n, dtype=np.bool_)
    result[valid] = prefix_max_end[idx[valid]] >= w_ends[valid]
    return result


def compute_geom_ok(starts, ends, is_plus, ref_pos, ref_plus, tolerance: int) -> np.ndarray:
    regions = _build_promoter_regions(ref_pos, ref_plus, tolerance)
    out = np.zeros(len(starts), dtype=np.bool_)
    pm = is_plus
    out[pm] = _vectorized_is_tp(starts[pm], ends[pm], *regions["+"])
    out[~pm] = _vectorized_is_tp(starts[~pm], ends[~pm], *regions["-"])
    return out


def greedy_incremental_cum_matched(
    starts: np.ndarray,
    ends: np.ndarray,
    is_plus: np.ndarray,
    scores: np.ndarray,
    ref_pos: np.ndarray,
    ref_plus: np.ndarray,
    tolerance: int,
    geom_ok: np.ndarray,
) -> np.ndarray:
    """Return cum_matched[k] = matches after processing top k preds by score (k=0..N)."""
    n_ref = len(ref_pos)
    n = len(starts)
    desc_order = np.argsort(-scores, kind="mergesort")

    plus_ref_idx = np.where(ref_plus)[0]
    minus_ref_idx = np.where(~ref_plus)[0]
    plus_pos = ref_pos[plus_ref_idx]
    minus_pos = ref_pos[minus_ref_idx]
    o_p = np.argsort(plus_pos)
    o_m = np.argsort(minus_pos)
    plus_pos_s = plus_pos[o_p]
    minus_pos_s = minus_pos[o_m]
    plus_idx_s = plus_ref_idx[o_p]
    minus_idx_s = minus_ref_idx[o_m]

    matched_ref = np.zeros(n_ref, dtype=np.bool_)
    cum_matched = np.zeros(n + 1, dtype=np.int64)
    nmatch = 0

    for step in range(n):
        idx = int(desc_order[step])
        if not geom_ok[idx]:
            cum_matched[step + 1] = nmatch
            continue

        ws = int(starts[idx])
        we = int(ends[idx])
        center = (ws + we) // 2

        if is_plus[idx]:
            lo_t = we + UPSTREAM_MIN - tolerance
            hi_t = ws + UPSTREAM_MAX + tolerance
            if lo_t > hi_t:
                cum_matched[step + 1] = nmatch
                continue
            lo = int(np.searchsorted(plus_pos_s, lo_t, side="left"))
            hi = int(np.searchsorted(plus_pos_s, hi_t, side="right"))
            pos_s = plus_pos_s
            idx_s = plus_idx_s
        else:
            lo_t = we - UPSTREAM_MAX - tolerance
            hi_t = ws - UPSTREAM_MIN + tolerance
            if lo_t > hi_t:
                cum_matched[step + 1] = nmatch
                continue
            lo = int(np.searchsorted(minus_pos_s, lo_t, side="left"))
            hi = int(np.searchsorted(minus_pos_s, hi_t, side="right"))
            pos_s = minus_pos_s
            idx_s = minus_idx_s

        best_j = -1
        best_d = 10**18
        for j in range(lo, hi):
            ridx = int(idx_s[j])
            if matched_ref[ridx]:
                continue
            T = int(pos_s[j])
            if is_plus[idx]:
                reg_s = T - UPSTREAM_MAX - tolerance
                reg_e = T - UPSTREAM_MIN + tolerance
            else:
                reg_s = T + UPSTREAM_MIN - tolerance
                reg_e = T + UPSTREAM_MAX + tolerance
            if ws >= reg_s and we <= reg_e:
                d = abs(center - T)
                if d < best_d:
                    best_d = d
                    best_j = j

        if best_j >= 0:
            matched_ref[int(idx_s[best_j])] = True
            nmatch += 1
        cum_matched[step + 1] = nmatch

    return cum_matched


def greedy_matched_window_indices_topk(
    starts: np.ndarray,
    ends: np.ndarray,
    is_plus: np.ndarray,
    scores: np.ndarray,
    ref_pos: np.ndarray,
    ref_plus: np.ndarray,
    tolerance: int,
    geom_ok: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Among the first k windows in global descending score order (mergesort tie-break),
    return the original row indices that win a 1:1 greedy match to an unused reference
    (same rules as greedy_incremental_cum_matched). len(output) == cum_matched[k].
    """
    n_ref = len(ref_pos)
    n = len(starts)
    k = int(min(max(k, 0), n))
    if k == 0:
        return np.array([], dtype=np.int64)

    desc_order = np.argsort(-scores, kind="mergesort")

    plus_ref_idx = np.where(ref_plus)[0]
    minus_ref_idx = np.where(~ref_plus)[0]
    plus_pos = ref_pos[plus_ref_idx]
    minus_pos = ref_pos[minus_ref_idx]
    o_p = np.argsort(plus_pos)
    o_m = np.argsort(minus_pos)
    plus_pos_s = plus_pos[o_p]
    minus_pos_s = minus_pos[o_m]
    plus_idx_s = plus_ref_idx[o_p]
    minus_idx_s = minus_ref_idx[o_m]

    matched_ref = np.zeros(n_ref, dtype=np.bool_)
    matched_window_idx: list[int] = []

    for step in range(k):
        idx = int(desc_order[step])
        if not geom_ok[idx]:
            continue

        ws = int(starts[idx])
        we = int(ends[idx])
        center = (ws + we) // 2

        if is_plus[idx]:
            lo_t = we + UPSTREAM_MIN - tolerance
            hi_t = ws + UPSTREAM_MAX + tolerance
            if lo_t > hi_t:
                continue
            lo = int(np.searchsorted(plus_pos_s, lo_t, side="left"))
            hi = int(np.searchsorted(plus_pos_s, hi_t, side="right"))
            pos_s = plus_pos_s
            idx_s = plus_idx_s
        else:
            lo_t = we - UPSTREAM_MAX - tolerance
            hi_t = ws - UPSTREAM_MIN + tolerance
            if lo_t > hi_t:
                continue
            lo = int(np.searchsorted(minus_pos_s, lo_t, side="left"))
            hi = int(np.searchsorted(minus_pos_s, hi_t, side="right"))
            pos_s = minus_pos_s
            idx_s = minus_idx_s

        best_j = -1
        best_d = 10**18
        for j in range(lo, hi):
            ridx = int(idx_s[j])
            if matched_ref[ridx]:
                continue
            T = int(pos_s[j])
            if is_plus[idx]:
                reg_s = T - UPSTREAM_MAX - tolerance
                reg_e = T - UPSTREAM_MIN + tolerance
            else:
                reg_s = T + UPSTREAM_MIN - tolerance
                reg_e = T + UPSTREAM_MAX + tolerance
            if ws >= reg_s and we <= reg_e:
                d = abs(center - T)
                if d < best_d:
                    best_d = d
                    best_j = j

        if best_j >= 0:
            matched_ref[int(idx_s[best_j])] = True
            matched_window_idx.append(idx)

    return np.array(matched_window_idx, dtype=np.int64)


def generate_pr_curve_one_to_one(
    starts, ends, is_plus, scores,
    ref_pos, ref_plus, tolerance: int,
    num_thresholds: int = 100,
):
    print(f"\n  One-to-one matching (tolerance={tolerance}) ...")
    t0 = time.time()
    geom_ok = compute_geom_ok(starts, ends, is_plus, ref_pos, ref_plus, tolerance)
    print(f"    Geometrically matchable windows: {geom_ok.sum():,} / {len(geom_ok):,}")

    print(f"  Greedy pass over {len(scores):,} windows (desc score) ...")
    t1 = time.time()
    cum_matched = greedy_incremental_cum_matched(
        starts, ends, is_plus, scores, ref_pos, ref_plus, tolerance, geom_ok,
    )
    print(f"    Greedy done in {time.time()-t1:.1f}s; final matches: {cum_matched[-1]:,}")

    desc_order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[desc_order]

    percentiles = np.linspace(0, 100, num_thresholds)
    thr_pct = np.percentile(scores, percentiles)
    p99 = float(np.percentile(scores, 99))
    s_max = float(scores.max())
    thr_high = np.linspace(s_max, p99, min(80, num_thresholds))
    thresholds = np.sort(np.unique(np.concatenate([thr_pct, thr_high])))[::-1]

    print(f"    {len(thresholds)} unique thresholds [{thresholds[-1]:.6f} .. {thresholds[0]:.6f}]")

    n_total_ref = len(ref_pos)
    thresholds_out, precisions_out, recalls_out, f1_out, npred_out = [], [], [], [], []

    for thr in thresholds:
        k = int(np.searchsorted(-sorted_scores, -thr, side="right"))
        if k == 0:
            thresholds_out.append(thr)
            precisions_out.append(0.0)
            recalls_out.append(0.0)
            f1_out.append(0.0)
            npred_out.append(0)
            continue

        matched = int(cum_matched[k])
        prec = matched / k
        rec = matched / n_total_ref if n_total_ref > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        thresholds_out.append(thr)
        precisions_out.append(prec)
        recalls_out.append(rec)
        f1_out.append(f1)
        npred_out.append(k)

    print(f"  Total PR (1:1) generation: {time.time()-t0:.1f}s")
    return thresholds_out, precisions_out, recalls_out, f1_out, npred_out


def plot_pr_curve(thresholds, precisions, recalls, output_file, title):
    sorted_idx = np.argsort(recalls)
    sorted_rec = np.array(recalls)[sorted_idx]
    sorted_pre = np.array(precisions)[sorted_idx]
    try:
        auprc = float(np.trapezoid(sorted_pre, sorted_rec))
    except AttributeError:
        auprc = float(np.trapz(sorted_pre, sorted_rec))
    print(f"  AUPRC: {auprc:.6f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(
        sorted_rec, sorted_pre, "b-", linewidth=CURVE_LINEWIDTH,
        label=f"PR Curve (AUPRC={auprc:.3f})", zorder=3,
    )
    ax.scatter(recalls, precisions, c="blue", s=20, alpha=0.3, edgecolors="none", zorder=2)

    f1_scores = [2*p*r/(p+r) if (p+r) > 0 else 0 for p, r in zip(precisions, recalls)]
    best_idx = int(np.argmax(f1_scores))
    best_f1 = f1_scores[best_idx]
    best_prec = precisions[best_idx]
    best_rec = recalls[best_idx]
    best_thr = thresholds[best_idx]
    ax.plot(best_rec, best_prec, "ro", markersize=12,
            label=f"Max F1: {best_f1:.3f} (T={best_thr:.4f})", zorder=5)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=LEGEND_FONTSIZE, framealpha=0.95)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    highres = output_file.replace(".png", "_highres.png")
    plt.savefig(highres, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_file}")
    print(f"  Saved: {highres}")
    return best_idx, best_f1, best_thr, best_prec, best_rec, auprc


def save_pr_data(thresholds, precisions, recalls, output_file, num_predictions=None):
    f1s = [2*p*r/(p+r) if (p+r) > 0 else 0 for p, r in zip(precisions, recalls)]
    data = {"threshold": thresholds, "precision": precisions, "recall": recalls, "f1_score": f1s}
    if num_predictions is not None:
        data["num_predictions"] = num_predictions
    pd.DataFrame(data).to_csv(output_file, sep="\t", index=False)
    print(f"  Saved: {output_file}")


def main():
    ap = argparse.ArgumentParser(
        description="PR curve with 1:1 window–promoter greedy matching")
    ap.add_argument("--input-tsv")
    ap.add_argument("--reference-csv")
    ap.add_argument("--tolerance", type=int, required=True)
    ap.add_argument("--num-thresholds", type=int, default=100)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--replot-only",
        action="store_true",
        help="Regenerate PNGs from existing pr_curve_data.tsv in output-dir (no full eval).",
    )
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.replot_only:
        data_file = os.path.join(args.output_dir, "pr_curve_data.tsv")
        if not os.path.isfile(data_file):
            raise SystemExit(f"--replot-only: missing {data_file}")
        df = pd.read_csv(data_file, sep="\t")
        thresholds = df["threshold"].tolist()
        precisions = df["precision"].tolist()
        recalls = df["recall"].tolist()
        tol_tag = f"{args.tolerance}bp" if args.tolerance > 0 else "no"
        title = (
            f"PR Curve: 1:1 Window–Promoter Matching (greedy, score order)\n"
            f"(Promoter=[TSS-{UPSTREAM_MAX},TSS-{UPSTREAM_MIN}], "
            f"tolerance={tol_tag}, strand-aware)"
        )
        plot_file = os.path.join(args.output_dir, "pr_curve.png")
        plot_pr_curve(thresholds, precisions, recalls, plot_file, title)
        print(f"  Replotted from {data_file} -> {plot_file}")
        return

    if not args.input_tsv or not args.reference_csv:
        raise SystemExit("--input-tsv and --reference-csv are required unless --replot-only")

    print("=" * 70)
    print("  PR CURVE — ONE-TO-ONE WINDOW / PROMOTER MATCHING")
    print("=" * 70)
    print(f"  Input TSV:    {args.input_tsv}")
    print(f"  Reference:    {args.reference_csv}")
    print(f"  Promoter def: [TSS-{UPSTREAM_MAX}, TSS-{UPSTREAM_MIN}]")
    print(f"  Window len:   {WINDOW_LEN} bp")
    print(f"  Tolerance:    ±{args.tolerance} bp")
    print(f"  Output:       {args.output_dir}")
    print()

    t_global = time.time()
    starts, ends, is_plus, scores = load_predictions(args.input_tsv)
    ref_pos, ref_plus = load_reference_tss(args.reference_csv)

    tol_tag = f"{args.tolerance}bp" if args.tolerance > 0 else "no"

    thresholds, precisions, recalls, f1s, npreds = generate_pr_curve_one_to_one(
        starts, ends, is_plus, scores,
        ref_pos, ref_plus, args.tolerance,
        num_thresholds=args.num_thresholds,
    )

    title = (
        f"PR Curve: 1:1 Window–Promoter Matching (greedy, score order)\n"
        f"(Promoter=[TSS-{UPSTREAM_MAX},TSS-{UPSTREAM_MIN}], "
        f"tolerance={tol_tag}, strand-aware)"
    )
    plot_file = os.path.join(args.output_dir, "pr_curve.png")
    _, best_f1, best_thr, best_prec, best_rec, auprc = plot_pr_curve(
        thresholds, precisions, recalls, plot_file, title,
    )

    data_file = os.path.join(args.output_dir, "pr_curve_data.tsv")
    save_pr_data(thresholds, precisions, recalls, data_file, num_predictions=npreds)

    elapsed = time.time() - t_global
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  AUPRC:         {auprc:.6f}")
    print(f"  Best F1:       {best_f1:.3f}  (T={best_thr:.6f}, P={best_prec:.3f}, R={best_rec:.3f})")
    print(f"  Reference TSS: {len(ref_pos):,}")
    print(f"  Total windows: {len(starts):,}")
    print(f"  Elapsed:       {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Output:        {args.output_dir}")


if __name__ == "__main__":
    main()
