#!/usr/bin/env bash
# setup.sh — verify GBactpro installation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -t 1 ]]; then
  GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
else
  GREEN=''; YELLOW=''; RED=''; NC=''
fi
ok() { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; FAILURES=$((FAILURES + 1)); }
FAILURES=0

echo "========================================"
echo " GBactpro setup check"
echo "========================================"

PYTHON_BIN="$(command -v python3 || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  fail "python3 not found"
else
  ok "python3: $PYTHON_BIN ($("$PYTHON_BIN" -c 'import sys; print(sys.version.split()[0])'))"
fi

check_py() {
  local mod=$1
  if "$PYTHON_BIN" -c "import $mod" 2>/dev/null; then
    ok "python module: $mod"
  else
    fail "python module missing: $mod (conda env create -f environment.yml && conda activate gbactpro)"
  fi
}

if [[ -n "$PYTHON_BIN" ]]; then
  check_py numpy
  check_py pandas
  check_py Bio
  check_py tensorflow
fi

check_model() {
  local dir=$1
  if [[ -f "$dir/saved_model.pb" && -f "$dir/metadata.json" && -d "$dir/variables" ]]; then
    ok "model present: $(basename "$dir")"
  else
    fail "model incomplete: $dir"
  fi
}

check_model "$SCRIPT_DIR/models/type1_35s10_random"
check_model "$SCRIPT_DIR/models/type1_pre_max29bp_random"

[[ -f "$SCRIPT_DIR/example/input/sequences.fasta" ]] \
  && ok "example FASTA present" \
  || fail "missing example/input/sequences.fasta"

[[ -x "$SCRIPT_DIR/scripts/gbactpro_predict.py" ]] || chmod +x "$SCRIPT_DIR/scripts/gbactpro_predict.py"
ok "scripts/gbactpro_predict.py executable"

echo ""
if [[ $FAILURES -eq 0 ]]; then
  echo -e "${GREEN}Setup OK. Next: ./test.sh${NC}"
  exit 0
fi
echo -e "${RED}$FAILURES check(s) failed.${NC}"
exit 1
