#!/usr/bin/env bash
# test.sh — smoke test for fixed-length prediction (+ optional genome quick demo)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GENOME_QUICK=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --genome-quick) GENOME_QUICK=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--genome-quick]"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -t 1 ]]; then
  GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
else
  GREEN=''; RED=''; YELLOW=''; NC=''
fi
pass() { echo -e "${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; FAILURES=$((FAILURES + 1)); }
info() { echo -e "${YELLOW}[INFO]${NC} $*"; }
FAILURES=0

echo "========================================"
echo " GBactpro test"
echo "========================================"

info "Pre-flight checks..."
bash "$SCRIPT_DIR/setup.sh" || { echo "setup.sh failed"; exit 1; }

TEST_OUT="$SCRIPT_DIR/test_output/predict"
rm -rf "$TEST_OUT"
mkdir -p "$TEST_OUT"

info "Running fixed-length prediction on example FASTA..."
python3 "$SCRIPT_DIR/scripts/gbactpro_predict.py" \
  -i "$SCRIPT_DIR/example/input/sequences.fasta" \
  -o "$TEST_OUT/predictions.tsv" \
  --model "$SCRIPT_DIR/models/type1_35s10_random" \
  -t 0.5

[[ -s "$TEST_OUT/predictions.tsv" ]] \
  && pass "predictions.tsv written" \
  || fail "predictions.tsv missing/empty"

NROWS=$(tail -n +2 "$TEST_OUT/predictions.tsv" | wc -l | tr -d ' ')
[[ "$NROWS" -eq 10 ]] \
  && pass "expected 10 prediction rows (got $NROWS)" \
  || fail "expected 10 rows, got $NROWS"

# Promoter examples should tend to score higher than non-promoter examples
AVG_POS=$(awk -F'\t' 'NR>1 && $1 ~ /^promoter_/ {s+=$4; n++} END{if(n) printf "%.4f", s/n; else print "nan"}' "$TEST_OUT/predictions.tsv")
AVG_NEG=$(awk -F'\t' 'NR>1 && $1 ~ /^nonpromoter_/ {s+=$4; n++} END{if(n) printf "%.4f", s/n; else print "nan"}' "$TEST_OUT/predictions.tsv")
info "mean score promoters=$AVG_POS  non-promoters=$AVG_NEG"
python3 - "$AVG_POS" "$AVG_NEG" <<'PY' && pass "promoter mean score > non-promoter mean score" || fail "score sanity check failed"
import sys
pos, neg = float(sys.argv[1]), float(sys.argv[2])
sys.exit(0 if pos > neg else 1)
PY

# Refresh golden example output for users
cp "$TEST_OUT/predictions.tsv" "$SCRIPT_DIR/example/output/predictions.tsv"
pass "updated example/output/predictions.tsv"

if [[ $GENOME_QUICK -eq 1 ]]; then
  info "Running genome-wide quick demo..."
  bash "$SCRIPT_DIR/paper/genome_wide/run.sh" --quick \
    || fail "genome-wide quick run failed"
fi

echo ""
echo "========================================"
if [[ $FAILURES -eq 0 ]]; then
  echo -e "${GREEN}All checks passed.${NC}"
  echo "  Predict output: $TEST_OUT/predictions.tsv"
  exit 0
fi
echo -e "${RED}$FAILURES check(s) failed.${NC}"
exit 1
