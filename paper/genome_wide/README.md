# Genome-wide reproduction (E. coli MG1655)

This module reproduces the paper’s **genome-wide scanning + evaluation** workflow for *Escherichia coli* MG1655 (`NC_000913.2`) using the **Type 1 pre_max29bp** model.

> This is **not** the same checkpoint as the v1 fixed-length predictor (`models/type1_35s10_random`, maxlen 31, post-pad).  
> Genome scan uses `models/type1_pre_max29bp_random` (maxlen **29**, trained with **pre**-padding on windows ≤29 bp).

## Pipeline steps

| Step | Output | Description |
|------|--------|-------------|
| 01 | `output/<mode>/01_data/` | Genome FASTA + TSS reference |
| 02 | `output/<mode>/02_model/` | Symlink to pre_max29bp SavedModel |
| 03 | `output/<mode>/03_scan/` | Double-strand all-window scores (BED + TSV) |
| 04 | `output/<mode>/04_eval_o2o_*` | One-to-one PR curve vs TSS promoters |
| 05 | `output/<mode>/05_summary/metrics.json` | Best F1 / AUPRC summary |

## Quick start

```bash
# From repo root (conda env active)
cd paper/genome_wide

# Smoke demo: first 100 kb only (NOT paper-comparable)
./run.sh --quick

# Full chromosome (slow; paper reproduction)
./run.sh --full
```

Optional: `--tolerance 100` expands promoter regions by ±100 bp (paper also reports 0 bp).

## Inputs

| File | Role |
|------|------|
| `data/NC_000913.2.fasta` | E. coli K-12 MG1655 genome |
| `data/ecoli_all4classes_tss_filtered.csv` | Reference TSS table (`Pos`, `Strand`, …) |
| `../../models/type1_pre_max29bp_random/` | Genome-scan model |

## How scanning works

1. Slide a window of length **L = 29** across the genome (step 1).
2. Score the **+** strand sequence and the **−** strand reverse-complement of the same interval.
3. Write every non-N window (`--all-windows`) with genomic coordinates and score.

Promoter geometry for evaluation (matching the paper):

- `+` strand TSS: promoter interval `[TSS−39, TSS−6]` (±tolerance)
- `−` strand TSS: promoter interval `[TSS+6, TSS+39]` (±tolerance)
- A predicted window is a geometric hit if fully contained and strand matches.
- **One-to-one** matching: greedy by descending score; each window and each TSS used at most once.

## Interpreting outputs

- `score`: model sigmoid probability (higher = more promoter-like)
- `pr_curve.png` / `pr_curve_data.tsv`: precision–recall under 1:1 matching
- `metrics.json`: AUPRC and max-F1 operating point

Quick mode metrics are for pipeline verification only. Use `--full` when comparing to paper tables/figures.
