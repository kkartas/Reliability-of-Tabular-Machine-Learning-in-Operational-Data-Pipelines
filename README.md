[![DOI](https://zenodo.org/badge/1194724607.svg)](https://doi.org/10.5281/zenodo.19296167)

# Reproducibility Instructions for Reviewers

This repository accompanies the paper:

`Reliability of Tabular Machine Learning in Operational Data Pipelines: Structured Corruption Stress Testing for Decision Support and Expert Systems`

This README is intentionally limited to the reproduction of the experiments reported in the paper.

## Requirements

- Python 3.11
- Windows PowerShell
- Internet access for downloading the pinned OpenML datasets

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements-lock.txt
```

`requirements-lock.txt` is the pinned environment for the reported experiments.

## Canonical Experiment Reproduction

Run the following commands from the repository root:

```powershell
$env:PYTHONHASHSEED='0'
$env:DATASET_SET='extended'
$env:DATASET_SOURCE_POLICY='openml_only'
$env:C1_ANCHOR_MODE='label_agnostic'
python run_experiments.py
```

These settings match the paper configuration:

- `DATASET_SET='extended'`: the 12 datasets used in the manuscript
- `DATASET_SOURCE_POLICY='openml_only'`: fail if a pinned OpenML source is unavailable
- `C1_ANCHOR_MODE='label_agnostic'`: manuscript setting for the C1 corruption family
- `PYTHONHASHSEED='0'`: fixed process-level hashing for deterministic behavior

## Main Outputs

A successful run writes the experiment outputs under `results/`, including:

- `results/raw/metrics_long.csv`
- `results/metrics_summary.csv`
- `results/raw/corruption_diagnostics_long.csv`
- `results/raw/policy_metrics_long.csv`
- `results/tables/policy_metrics_summary.csv`
- `results/raw/calibration_metrics_long.csv`
- `results/tables/calibration_metrics_summary.csv`
- `results/run_metadata.json`

## Notes

- The canonical reproduction path is `python run_experiments.py`. No other step is required to reproduce the experiments themselves.
- The run downloads the pinned OpenML datasets and regenerates all experiment outputs locally.
- If a pinned OpenML dataset is unavailable, the run fails by design instead of silently falling back to another source.
