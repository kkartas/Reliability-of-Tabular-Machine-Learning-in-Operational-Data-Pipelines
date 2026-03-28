# Dataset Inclusion Protocol

Date: 2026-02-25  
Protocol ID: DSP-20260225-01  
Scope: Phase-2 dataset expansion (target: 8-12 binary datasets total)

## 1) Objective
Expand dataset coverage while preserving:
- comparability across corruption experiments,
- reproducibility (pinned OpenML identifiers),
- valid binary target semantics,
- leakage-safe feature/target handling.

## 2) Eligibility Criteria (Must Pass All)

### 2.1 Structural Criteria
- Binary target after harmonization (`n_classes == 2`).
- At least 5,000 rows (preferred) or explicit waiver documented.
- At least 8 features total.
- At least one numerical and one categorical feature after harmonization (preferred).

### 2.2 Label Quality Criteria
- Positive class prevalence in [0.05, 0.95] (outside range requires justification).
- Target mapping documented explicitly (for example, `{1,2} -> positive=2`).
- No ambiguous multi-label or probabilistic target columns.

### 2.3 Leakage and Integrity Criteria
- No direct target proxies among features (manual + heuristic check).
- No timestamp-derived leakage fields unless transformed with leakage-safe policy.
- Missingness and dtype profile recorded.

### 2.4 Reproducibility Criteria
- Prefer OpenML `data_id` over name/version.
- Manuscript mode uses `DATASET_SOURCE_POLICY='openml_only'` (fail-fast if pinned OpenML fetch fails).
- CSV fallback is permitted only in explicit engineering mode (`DATASET_SOURCE_POLICY='openml_or_csv'`) and must include schema note, target-column mapping, and source file SHA256 in metadata.
- Dataset profile must be emitted in run metadata and dataset profile table.

## 3) Selection Procedure
1. Candidate discovery from OpenML/UCI benchmarks.
2. Preliminary fetch and schema parse.
3. Automated checks:
   - binary target check,
   - class prevalence check,
   - feature count and dtype mix check.
4. Manual leakage review.
5. Assign inclusion decision:
   - `include`,
   - `include_with_waiver`,
   - `exclude`.
6. Record decision in dataset registry changelog.

## 4) Required Artifacts Per Included Dataset
- Pinned identifier (`data_id` or local path).
- Target mapping rule.
- Feature schema summary:
  - row count,
  - feature count,
  - numeric/categorical counts,
  - prevalence.
- Inclusion note (why included, known caveats).

## 5) Expansion Hypotheses (for broader benchmark)
- H-D1: C2 remains the dominant single corruption in average AUC/ECE degradation across expanded datasets.
- H-D2: compound C2+C4 remains stronger than median single-corruption effects.
- H-D3: calibration mitigation remains heterogeneous across datasets even with broader coverage.

These are benchmark-level hypotheses and will be reported as confirmatory only after power upgrade (WS6.2).

## 6) Initial Candidate Families (to evaluate next)
- Credit risk / default
- Marketing response / churn
- Income and socio-economic classification
- Medical risk prediction
- Fraud/anomaly binary classification

## 7) Exclusion Reasons (must be explicit)
- non-binary task,
- irreproducible source identifier,
- severe leakage risk,
- unusable schema quality,
- licensing/access constraints.
