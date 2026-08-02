#!/usr/bin/env bash
# Run the v1 predict CLI in a container (Docker, or udocker on HPC without docker group).
#
# Usage (from repo root):
#   ./scripts/container_predict.sh
#   ./scripts/container_predict.sh -i example/input/sequences.fasta -o results/predictions.tsv
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

IMAGE="${GBACTPRO_IMAGE:-ghcr.io/ximenahan/gbactpro:latest}"
LOCAL_TAG="${GBACTPRO_LOCAL_TAG:-gbactpro:local}"

IN_HOST="${ROOT}/example/input/sequences.fasta"
OUT_HOST="${ROOT}/results/predictions.tsv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input) IN_HOST="$2"; shift 2 ;;
    -o|--output) OUT_HOST="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [-i fasta] [-o tsv]"
      echo "Env: GBACTPRO_IMAGE (default $IMAGE)"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Resolve relative paths from repo root
[[ "$IN_HOST" != /* ]] && IN_HOST="$ROOT/$IN_HOST"
[[ "$OUT_HOST" != /* ]] && OUT_HOST="$ROOT/$OUT_HOST"
mkdir -p "$(dirname "$OUT_HOST")"

IN_DIR="$(cd "$(dirname "$IN_HOST")" && pwd)"
OUT_DIR="$(cd "$(dirname "$OUT_HOST")" && pwd)"
IN_BASE="$(basename "$IN_HOST")"
OUT_BASE="$(basename "$OUT_HOST")"

have_docker() { command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; }
have_udocker() { command -v udocker >/dev/null 2>&1; }

run_predict() {
  # $1 = runner name for logs; remaining invoked as container command prefix... 
  :
}

if have_docker; then
  if docker image inspect "$LOCAL_TAG" >/dev/null 2>&1; then
    IMG="$LOCAL_TAG"
  else
    IMG="$IMAGE"
    if ! docker image inspect "$IMG" >/dev/null 2>&1; then
      echo "[container_predict] pulling $IMG ..."
      docker pull "$IMG"
    fi
  fi
  echo "[container_predict] docker → $IMG"
  docker run --rm \
    -v "$IN_DIR:/in:ro" \
    -v "$OUT_DIR:/out" \
    "$IMG" \
    -i "/in/$IN_BASE" \
    -o "/out/$OUT_BASE"
  echo "[container_predict] wrote $OUT_HOST"
  exit 0
fi

if have_udocker; then
  export PATH="${HOME}/.local/bin:${PATH}"
  echo "[container_predict] docker daemon not usable; using udocker → $IMAGE"
  udocker install >/dev/null 2>&1 || true
  CONT="gbactpro_predict"
  REPO_NAME="$(udocker images 2>/dev/null | awk 'BEGIN{IGNORECASE=1} /gbactpro/ {print $1; exit}')"
  if [[ -z "${REPO_NAME:-}" ]]; then
    echo "[container_predict] pulling $IMAGE (first time) ..."
    udocker pull "$IMAGE"
    REPO_NAME="$(udocker images | awk 'BEGIN{IGNORECASE=1} /gbactpro/ {print $1; exit}')"
  fi
  if [[ -z "${REPO_NAME:-}" ]]; then
    echo "ERROR: udocker could not find a pulled gbactpro image."
    echo "If GHCR is private, make the package public (GitHub → Packages → package settings)."
    exit 1
  fi
  # Create once; reuse afterwards (udocker has no `ps -a`)
  if ! udocker ps 2>/dev/null | grep -Fq "$CONT"; then
    udocker create --name="$CONT" "$REPO_NAME" >/dev/null
  fi
  # Image ENTRYPOINT is already gbactpro_predict.py — pass CLI args only.
  udocker run \
    --volume="$IN_DIR:/in" \
    --volume="$OUT_DIR:/out" \
    "$CONT" \
    -i "/in/$IN_BASE" \
    -o "/out/$OUT_BASE"
  echo "[container_predict] wrote $OUT_HOST"
  exit 0
fi

cat <<EOF
ERROR: neither Docker nor udocker is usable.

On this host your account is not in the 'docker' group and cannot use sudo.
Ask an administrator (or a user with sudo) to run:

  sudo usermod -aG docker $USER
  # then log out and back in (or: newgrp docker)

Until then, use Option A (conda) from the README — fully supported.
Optional HPC fallback after the image is published:

  pip install --user udocker && udocker install
  ./scripts/container_predict.sh
EOF
exit 1
