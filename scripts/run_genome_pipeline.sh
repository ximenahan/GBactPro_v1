#!/usr/bin/env bash
# Thin wrapper → paper/genome_wide/run.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/paper/genome_wide/run.sh" "$@"
