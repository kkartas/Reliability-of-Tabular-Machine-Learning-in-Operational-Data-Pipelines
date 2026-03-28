# Statistical Analysis Plan (Phase 1)

Date: 2026-02-25  
Plan ID: SAP-20260225-01  
Scope: Current manuscript configuration (12 datasets, 3 models, 2 encodings, 20 seeds)

## 1) Endpoint Taxonomy

### 1.1 Primary Endpoints (Confirmatory)
Primary endpoints are severe-versus-baseline deltas computed on uncalibrated outputs:
- Delta AUC
- Delta F1@0.5
- Delta ECE

Primary comparison settings:
- C2 at severity 0.4 vs 0.0
- C2+C4 at (0.4, 0.4) vs (0.0, 0.0)

Aggregation level for primary interpretation:
- Cross-dataset unweighted mean of per-dataset deltas (with bootstrap CIs).
- Sensitivity: dataset-row-weighted means.
- C3 aggregation excludes datasets with zero categorical columns.

### 1.2 Secondary Endpoints (Supportive)
- Same delta metrics for C1, C3, and C4.
- Temperature-scaled counterparts.
- Encoding ablation (`unknown_bucket` minus `ignore_unknown`) under C3 severity 0.4.

### 1.3 Exploratory Endpoints
- Per-setting paired tests across all dataset-model-encoding combinations.
- Metric-level BH-controlled discovery profiles by corruption family.

## 2) Hypothesis Families

### Family F1 (Primary, Confirmatory)
For each metric in {AUC, F1, ECE}:
- H1a: C2 severe-vs-baseline effect is non-zero and directionally harmful.
- H1b: C2+C4 severe-vs-baseline effect is non-zero and directionally harmful.

Multiplicity control:
- BH correction applied within metric family for confirmatory tests.
- With 20 paired seeds, setting-level stability is improved; cross-dataset confirmatory scope remains bounded by 12 datasets.

### Family F2 (Secondary, Supportive)
- Directional comparisons across C1, C3, C4, and C2.
- Compound-vs-single qualitative comparisons.

Multiplicity control:
- Report corrected and uncorrected p-values; interpret as supportive, not definitive.

### Family F3 (Exploratory)
- Full paired-test map over all settings.

Multiplicity control:
- BH within each metric over exploratory grid.
- Interpret as pattern discovery.

## 3) Effect Size and Uncertainty Reporting
- For means over seeds: Student-t 95% CIs (df=n-1).
- For delta effects: bootstrap 95% CI of mean delta.
- For paired tests: report t-statistic, p-value, BH-adjusted p-value, and dz effect size.
- For Wilcoxon tests: report both total paired count and non-zero effective paired count (zero-method transparency).
- For policy and calibration severe comparisons: report delta means with bootstrap 95% CIs against fixed/uncalibrated baselines.
- For policy/calibration pairwise contrasts: use dataset-cluster means as the inferential unit (average setting-level deltas within dataset, then paired tests across datasets).
- For policy pairwise and calibration pairwise batteries: apply BH once over the full battery in each table (all listed scenarios and metrics).

## 4) Decision Rules for Manuscript Claims
- Confirmatory language is restricted to predefined Family F1 within benchmark scope and requires concordant effect sizes, CIs, and corrected p/q values.
- Secondary language ("consistent with", "suggestive") for Family F2.
- Exploratory language ("pattern", "screening-level evidence") for Family F3.

## 5) Power and Scope Status
- Repetition budget has been upgraded to 20 paired seeds for the manuscript rerun.
- Remaining inferential bound is dataset count (n=12); broader cross-dataset confirmation requires expanded benchmark coverage.
- Sensitivity calculation (paired two-sided t-test, alpha=0.05, target power=0.80, n=12 dataset-level units) gives minimum detectable standardized paired effect approximately dz=0.89.
