#!/usr/bin/env bash
# paper/genome_wide/run.sh — E. coli genome-wide scan + one-to-one evaluation
#
# Usage:
#   ./run.sh --quick          # ~100 kb demo region (CI / smoke)
#   ./run.sh --full           # full NC_000913.2 (paper reproduction; slow)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

MODE=""
TOLERANCE=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --quick) MODE=quick; shift ;;
    --full) MODE=full; shift ;;
    --tolerance) TOLERANCE="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$MODE" ]]; then
  echo "Usage: $0 --quick | --full [--tolerance 0]"
  exit 1
fi

GENOME="$SCRIPT_DIR/data/NC_000913.2.fasta"
TSS_CSV="$SCRIPT_DIR/data/ecoli_all4classes_tss_filtered.csv"
MODEL="$REPO_ROOT/models/type1_pre_max29bp_random"
OUT_ROOT="$SCRIPT_DIR/output/${MODE}"
SCAN_DIR="$OUT_ROOT/03_scan"
EVAL_DIR="$OUT_ROOT/04_eval_o2o_tol${TOLERANCE}"
SUMMARY="$OUT_ROOT/05_summary/metrics.json"

mkdir -p "$OUT_ROOT/01_data" "$OUT_ROOT/02_model" "$SCAN_DIR" "$EVAL_DIR" "$(dirname "$SUMMARY")"

# Step 01 — data check
echo "[01] Checking genome + TSS reference..."
[[ -f "$GENOME" ]] || { echo "ERROR: missing $GENOME"; exit 1; }
[[ -f "$TSS_CSV" ]] || { echo "ERROR: missing $TSS_CSV"; exit 1; }
ln -sfn "$GENOME" "$OUT_ROOT/01_data/NC_000913.2.fasta"
ln -sfn "$TSS_CSV" "$OUT_ROOT/01_data/ecoli_all4classes_tss_filtered.csv"

# Step 02 — model check
echo "[02] Checking genome-scan model (pre_max29bp)..."
[[ -f "$MODEL/saved_model.pb" ]] || { echo "ERROR: missing model $MODEL"; exit 1; }
ln -sfn "$MODEL" "$OUT_ROOT/02_model/type1_pre_max29bp_random"

REGION_ARGS=()
PREFIX="NC_000913.2_pre_max29bp"
if [[ "$MODE" == "quick" ]]; then
  # 100 kb demo around a gene-dense early genome region
  REGION_ARGS=(--region-start 0 --region-end 100000)
  PREFIX="NC_000913.2_pre_max29bp_quick100kb"
  echo "[03] Scanning quick region 0–100000 (both strands, all windows)..."
else
  echo "[03] Scanning FULL genome (both strands, all windows) — this is slow..."
fi

export SCAN_INFER_BATCH_SIZE="${SCAN_INFER_BATCH_SIZE:-1024}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

python3 "$SCRIPT_DIR/scripts/genome_scan.py" \
  --genome "$GENOME" \
  --model "$MODEL" \
  --all-windows \
  --out-dir "$SCAN_DIR" \
  --prefix "$PREFIX" \
  --padding post \
  "${REGION_ARGS[@]}"

TSV="$SCAN_DIR/${PREFIX}_allwindows.tsv"
[[ -s "$TSV" ]] || { echo "ERROR: scan TSV not produced: $TSV"; ls -la "$SCAN_DIR" || true; exit 1; }
echo "  Scan TSV: $TSV"

# Step 04 — one-to-one PR evaluation
# Quick mode: restrict TSS reference to the scanned region so metrics are locally meaningful.
EVAL_TSS="$TSS_CSV"
if [[ "$MODE" == "quick" ]]; then
  EVAL_TSS="$OUT_ROOT/01_data/tss_in_region.csv"
  python3 - "$TSS_CSV" "$EVAL_TSS" 0 100000 <<'PY'
import sys
import pandas as pd

src, dst, start, end = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
df = pd.read_csv(src)
pos = df["Pos"].astype(int)
plus = df["Strand"].astype(str) == "+"
reg_s = pos.copy()
reg_e = pos.copy()
# + strand promoter [TSS-39, TSS-6]; - strand [TSS+6, TSS+39]
reg_s.loc[plus] = pos[plus] - 39
reg_e.loc[plus] = pos[plus] - 6
reg_s.loc[~plus] = pos[~plus] + 6
reg_e.loc[~plus] = pos[~plus] + 39
overlap = (reg_s < end) & (reg_e > start)
sub = df.loc[overlap].copy()
sub.to_csv(dst, index=False)
print("  Quick TSS subset: {} / {} (region [{}, {}))".format(len(sub), len(df), start, end))
PY
fi

echo "[04] One-to-one PR evaluation (tolerance=${TOLERANCE} bp)..."
python3 "$SCRIPT_DIR/scripts/evaluate_scan_o2o.py" \
  --input-tsv "$TSV" \
  --reference-csv "$EVAL_TSS" \
  --tolerance "$TOLERANCE" \
  --num-thresholds 50 \
  --output-dir "$EVAL_DIR"

# Step 05 — summary JSON
echo "[05] Writing summary..."
python3 - "$EVAL_DIR" "$SUMMARY" "$MODE" "$TOLERANCE" "$TSV" <<'PY'
import json, sys
from pathlib import Path
import pandas as pd

eval_dir, summary_path, mode, tol, tsv = sys.argv[1:6]
pr = Path(eval_dir) / "pr_curve_data.tsv"
df = pd.read_csv(pr, sep="\t")
f1 = df["f1_score"] if "f1_score" in df.columns else (2 * df["precision"] * df["recall"] / (df["precision"] + df["recall"]).replace(0, float("nan")))
best_i = int(f1.idxmax())
# AUPRC via trapezoid on recall-sorted curve
sdf = df.sort_values("recall")
try:
    import numpy as np
    auprc = float(np.trapz(sdf["precision"], sdf["recall"]))
except Exception:
    auprc = None
metrics = {
    "mode": mode,
    "tolerance_bp": int(tol),
    "scan_tsv": tsv,
    "n_thresholds": int(len(df)),
    "best_f1": float(f1.iloc[best_i]),
    "best_threshold": float(df.loc[best_i, "threshold"]),
    "best_precision": float(df.loc[best_i, "precision"]),
    "best_recall": float(df.loc[best_i, "recall"]),
    "auprc": auprc,
    "pr_curve_data": str(pr),
    "pr_curve_png": str(Path(eval_dir) / "pr_curve.png"),
    "note": "Quick mode uses a 100 kb region; metrics are NOT paper-comparable. Use --full for reproduction.",
}
Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
Path(summary_path).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
print(json.dumps(metrics, indent=2))
PY

echo ""
echo "Done. Summary: $SUMMARY"
echo "PR curve:     $EVAL_DIR/pr_curve.png"
