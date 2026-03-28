from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"


def _bootstrap_mean_ci(values: np.ndarray, seed: int = 2042, n_boot: int = 5000) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    if len(values) == 1:
        v = float(values[0])
        return (v, v)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = samples.mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _cluster_bootstrap_ci(
    dataset_means: np.ndarray, seed: int, n_boot: int = 10000
) -> Tuple[float, float]:
    dataset_means = np.asarray(dataset_means, dtype=float)
    dataset_means = dataset_means[np.isfinite(dataset_means)]
    if len(dataset_means) == 0:
        return (np.nan, np.nan)
    if len(dataset_means) == 1:
        v = float(dataset_means[0])
        return (v, v)
    rng = np.random.default_rng(seed)
    samples = rng.choice(
        dataset_means, size=(n_boot, len(dataset_means)), replace=True
    )
    means = samples.mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _fmt(x: float, digits: int = 4, signed: bool = False) -> str:
    if pd.isna(x):
        return "--"
    if signed:
        return f"{x:+.{digits}f}"
    return f"{x:.{digits}f}"


def _fmt_ci(mean: float, low: float, high: float, digits: int = 4, signed: bool = True) -> str:
    return f"{_fmt(mean, digits=digits, signed=signed)} [{_fmt(low, digits=digits, signed=signed)}, {_fmt(high, digits=digits, signed=signed)}]"


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _write(path: Path, lines: Iterable[str]) -> None:
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")


def _dataset_profiles_df() -> pd.DataFrame:
    path = TABLES / "table_dataset_profiles.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _c3_eligible_datasets() -> set[str]:
    df = _dataset_profiles_df()
    if df.empty or "dataset" not in df.columns:
        return set()
    if "n_cat_cols" not in df.columns:
        return set(df["dataset"].astype(str).tolist())
    return {
        str(row["dataset"])
        for _, row in df.iterrows()
        if pd.notna(row["n_cat_cols"]) and int(row["n_cat_cols"]) > 0
    }


def build_dataset_profiles() -> None:
    df = pd.read_csv(TABLES / "table_dataset_profiles.csv")
    pretty = {
        "adult_income": "adult",
        "airlines": "airlines",
        "aps_failure": "APS failure",
        "bank_marketing": "bank mkt",
        "compass": "compass",
        "credit_default": "credit default",
        "diabetes_hospitals_fairlearn": "diabetes hospitals",
        "diabetes130us": "diab130us",
        "electricity": "electricity",
        "kick": "kick",
        "law_school_admission": "law admission",
        "telco_churn": "telco churn",
    }
    lines = [
        r"\begin{tabular}{lrrrrrrl}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{OpenML ID} & \textbf{Rows} & \textbf{Features} & \textbf{Num./Cat.} & \textbf{Positive rate} & \textbf{Max col. missing} & \textbf{Prov.} \\",
        r"\midrule",
    ]
    for _, row in df.sort_values("dataset").iterrows():
        source_policy = str(row.get("source_policy", "")).strip()
        source_sha = str(row.get("source_sha256", "")).strip()
        if len(source_sha) > 18:
            source_sha = source_sha[:12] + "..."
        source_policy_short = source_policy
        if source_policy == "openml_only":
            source_policy_short = "openml"
        elif source_policy == "openml_or_csv":
            source_policy_short = "openml/csv"

        if source_sha.startswith("N/A"):
            provenance = source_policy_short
        else:
            provenance = f"{source_policy_short}/{source_sha}".strip(" /")
        lines.append(
            f"{_latex_escape(pretty.get(row['dataset'], row['dataset']))} & "
            f"{int(row['openml_data_id'])} & {int(row['n_rows'])} & {int(row['n_features'])} & "
            f"{int(row['n_num_cols'])}/{int(row['n_cat_cols'])} & {_fmt(row['positive_rate'], digits=3)} & "
            f"{_fmt(row['max_col_missing_rate'], digits=3)} & {_latex_escape(provenance)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write(TABLES / "table_dataset_profiles.tex", lines)


def build_dataset_selection_audit() -> None:
    profiles = _dataset_profiles_df()
    if profiles.empty:
        return

    included = set(profiles["dataset"].astype(str).tolist())
    core_aliases = {"adult_income", "bank_marketing", "credit_default"}
    waiver_notes = {
        "airlines": "Included with waiver: minimum feature count (7 < default 8).",
        "diabetes130us": "Included with waiver: minimum feature count and categorical-column requirement.",
        "aps_failure": "Included with waiver: categorical-column requirement and positive-rate lower bound.",
    }

    screened = []
    screening_files = [
        ROOT / "docs" / "dataset_screening_20260225.csv",
        ROOT / "docs" / "dataset_screening_20260226.csv",
    ]
    for path in screening_files:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["screening_source"] = path.name
        screened.append(df)

    screened_df = pd.concat(screened, ignore_index=True) if screened else pd.DataFrame()
    if not screened_df.empty:
        screened_df = screened_df.sort_values("screening_source").drop_duplicates(
            subset=["alias"], keep="last"
        )
    screened_aliases = (
        set(screened_df["alias"].astype(str).tolist()) if not screened_df.empty else set()
    )

    all_aliases = sorted(included | screened_aliases | core_aliases)
    rows = []
    for alias in all_aliases:
        prof_row = profiles[profiles["dataset"].astype(str) == alias]
        if not prof_row.empty:
            data_id = int(prof_row.iloc[0]["openml_data_id"])
        elif not screened_df.empty and alias in screened_aliases:
            data_id = int(screened_df.loc[screened_df["alias"] == alias, "data_id"].iloc[0])
        else:
            data_id = -1

        if alias in included:
            if alias in waiver_notes:
                decision = "include_with_waiver"
                reason = waiver_notes[alias]
            else:
                decision = "include"
                reason = "Included after protocol checks."
        else:
            decision = "exclude"
            if not screened_df.empty and alias in screened_aliases:
                srow = screened_df[screened_df["alias"] == alias].iloc[0]
                n_cat = int(srow.get("n_cat_cols", 0))
                if n_cat == 0:
                    reason = "Excluded: no categorical columns under protocol defaults and no waiver."
                else:
                    reason = "Excluded from final 12-dataset scope after screening."
            else:
                reason = "Not part of finalized benchmark scope."

        if alias in core_aliases:
            source = "core"
        elif not screened_df.empty and alias in screened_aliases:
            raw_source = str(
                screened_df.loc[screened_df["alias"] == alias, "screening_source"].iloc[0]
            )
            source = raw_source.replace("dataset_screening_", "screen-").replace(".csv", "")
        else:
            source = "N/A"

        rows.append(
            {
                "dataset": alias,
                "openml_data_id": data_id if data_id >= 0 else np.nan,
                "decision": decision,
                "screening_source": source,
                "reason": reason,
            }
        )

    out = pd.DataFrame(rows).sort_values(["decision", "dataset"]).reset_index(drop=True)
    out.to_csv(TABLES / "table_dataset_selection_audit.csv", index=False)

    lines = [
        r"\begin{tabularx}{\textwidth}{XrllX}",
        r"\toprule",
        r"\textbf{Dataset alias} & \textbf{OpenML ID} & \textbf{Decision} & \textbf{Source} & \textbf{Reason} \\",
        r"\midrule",
    ]
    for _, row in out.iterrows():
        openml = "--" if pd.isna(row["openml_data_id"]) else str(int(row["openml_data_id"]))
        lines.append(
            f"{_latex_escape(row['dataset'])} & {openml} & {_latex_escape(row['decision'])} & "
            f"{_latex_escape(row['screening_source'])} & {_latex_escape(row['reason'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    _write(TABLES / "table_dataset_selection_audit.tex", lines)


def build_cross_dataset_effects() -> None:
    df = pd.read_csv(TABLES / "table_cross_dataset_effects.csv")
    order = ["C1", "C2", "C3", "C4", "C2+C4 (0.4,0.4)"]
    metrics = ["auc", "f1", "ece"]
    mlabel = {"auc": "AUC", "f1": "F1", "ece": "ECE"}
    lines = [
        r"\begin{tabularx}{\textwidth}{lcr>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X}",
        r"\toprule",
        r"\textbf{Corruption} & \textbf{Eligible $n$} & \textbf{Aggregation} & \textbf{Delta AUC (95\% CI)} & \textbf{Delta F1 (95\% CI)} & \textbf{Delta ECE (95\% CI)} \\",
        r"\midrule",
    ]
    for corr in order:
        sub = df[df["corruption"] == corr].copy()
        if sub.empty:
            continue
        n_eligible = int(sub["n_datasets"].max())
        agg = str(sub["aggregation"].iloc[0]).replace("_", " ")
        parts = {m: sub[sub["metric"] == m] for m in metrics}
        cells = []
        for m in metrics:
            row = parts[m].iloc[0]
            cells.append(
                _fmt_ci(
                    row["delta_mean_across_datasets"],
                    row["delta_ci95_low_boot"],
                    row["delta_ci95_high_boot"],
                    digits=4,
                    signed=True,
                )
            )
        corr_label = corr
        if corr == "C3":
            corr_label = "C3 (categorical-eligible only)"
        lines.append(f"{corr_label} & {n_eligible} & {agg} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    _write(TABLES / "table_cross_dataset_effects.tex", lines)


def build_primary_inference() -> None:
    df = pd.read_csv(TABLES / "table_primary_inference_summary.csv")
    mlabel = {"auc": "AUC", "f1": "F1", "ece": "ECE"}
    lines = [
        r"\begin{tabularx}{\textwidth}{llr>{\raggedright\arraybackslash}Xrcc}",
        r"\toprule",
        r"\textbf{Corruption} & \textbf{Metric} & \textbf{$n_{\mathrm{settings}}$} & \textbf{Delta (95\% CI)} & \textbf{Mean $d_z$} & \textbf{t-BH sig.} & \textbf{W-BH sig.} \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        delta_ci = _fmt_ci(
            row["delta_mean_across_settings"],
            row["delta_ci95_low_boot"],
            row["delta_ci95_high_boot"],
            digits=4,
            signed=True,
        )
        lines.append(
            f"{row['corruption']} & {mlabel[row['metric']]} & {int(row['n_settings'])} & {delta_ci} & {_fmt(row['effect_size_dz_mean'], 3, True)} & "
            f"{int(row['n_sig_t_bh_family_metric'])}/{int(row['n_settings'])} & "
            f"{int(row['n_sig_wilcoxon_bh_family_metric'])}/{int(row['n_settings'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    _write(TABLES / "table_primary_inference_summary.tex", lines)


def build_primary_dataset_inference() -> None:
    df = pd.read_csv(TABLES / "table_primary_dataset_inference_summary.csv")
    mlabel = {"auc": "AUC", "f1": "F1", "ece": "ECE"}
    lines = [
        r"\begin{tabularx}{\textwidth}{llr>{\raggedright\arraybackslash}Xcc}",
        r"\toprule",
        r"\textbf{Corruption} & \textbf{Metric} & \textbf{$n_{\mathrm{datasets}}$} & \textbf{Delta (95\% CI)} & \textbf{Harmful} & \textbf{Median $d_z$} \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        delta_ci = _fmt_ci(
            row["delta_mean_across_datasets"],
            row["delta_ci95_low_boot"],
            row["delta_ci95_high_boot"],
            digits=4,
            signed=True,
        )
        harmful = f"{int(row['harmful_direction_count'])}/{int(row['n_datasets'])}"
        lines.append(
            f"{row['corruption']} & {mlabel[row['metric']]} & {int(row['n_datasets'])} & {delta_ci} & "
            f"{harmful} & {_fmt(row['effect_size_dz_median'], 3, True)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    _write(TABLES / "table_primary_dataset_inference_summary.tex", lines)


def build_primary_hypothesis_tests() -> None:
    path = TABLES / "table_primary_hypothesis_tests.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return

    mlabel = {"auc": "AUC", "f1": "F1", "ece": "ECE"}
    lines = [
        r"\begin{tabularx}{\textwidth}{llrr>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}Xr}",
        r"\toprule",
        r"\textbf{Corruption} & \textbf{Metric} & \textbf{$n$} & \textbf{$n_{\neq 0}$} & \textbf{Delta (95\% CI)} & \textbf{t p / q(BH)} & \textbf{W p / q(BH)} & \textbf{Harmful direction} & \textbf{Median $d_z$} \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        delta_ci = _fmt_ci(
            row["delta_mean"],
            row["delta_ci95_low_boot"],
            row["delta_ci95_high_boot"],
            digits=4,
            signed=True,
        )
        t_cell = (
            f"{_fmt(row['t_p_value'], 4)} / {_fmt(row['t_p_value_bh_primary6'], 4)}"
            if pd.notna(row["t_p_value"])
            else "-- / --"
        )
        w_cell = (
            f"{_fmt(row['wilcoxon_p_value'], 4)} / {_fmt(row['wilcoxon_p_value_bh_primary6'], 4)}"
            if pd.notna(row["wilcoxon_p_value"])
            else "-- / --"
        )
        harmful = f"{int(row['harmful_direction_count'])}/{int(row['n_datasets'])}"
        lines.append(
            f"{row['corruption']} & {mlabel[str(row['metric'])]} & "
            f"{int(row['n_datasets'])} & {int(row['n_datasets_nonzero'])} & {delta_ci} & "
            f"{t_cell} & {w_cell} & {harmful} & {_fmt(row['paired_effect_size_dz_median'], 3, True)} \\\\"
        )

    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    _write(TABLES / "table_primary_hypothesis_tests.tex", lines)


def build_primary_hierarchical_sensitivity() -> None:
    path = TABLES / "table_primary_hierarchical_sensitivity.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return

    mlabel = {"auc": "AUC", "f1": "F1", "ece": "ECE"}
    lines = [
        r"\begin{tabularx}{\textwidth}{llr>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}Xc}",
        r"\toprule",
        r"\textbf{Corruption} & \textbf{Metric} & \textbf{$n_{ds}$} & \textbf{Naive CI (dataset bootstrap)} & \textbf{Hierarchical CI (cluster bootstrap)} & \textbf{Direction unchanged} \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        naive = _fmt_ci(
            row["delta_mean_across_datasets"],
            row["naive_ci95_low_boot"],
            row["naive_ci95_high_boot"],
            digits=4,
            signed=True,
        )
        hier = _fmt_ci(
            row["delta_mean_across_datasets"],
            row["hierarchical_ci95_low_boot"],
            row["hierarchical_ci95_high_boot"],
            digits=4,
            signed=True,
        )
        unchanged = "yes" if str(row.get("headline_direction_unchanged", "")).lower() == "true" else "no"
        lines.append(
            f"{row['corruption']} & {mlabel[str(row['metric'])]} & {int(row['n_datasets'])} & "
            f"{naive} & {hier} & {unchanged} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    _write(TABLES / "table_primary_hierarchical_sensitivity.tex", lines)


def _policy_delta_arrays(
    policy_summary: pd.DataFrame, corruption: str, scenario: str, c3_eligible: set[str] | None = None
) -> Dict[str, Dict[str, np.ndarray]]:
    if corruption == "C2+C4":
        sub = policy_summary[
            (policy_summary["corruption"] == "C2+C4")
            & (policy_summary["severity_c2"] == 0.4)
            & (policy_summary["severity_c4"] == 0.4)
            & (policy_summary["variant"] == "uncal")
        ].copy()
    else:
        sub = policy_summary[
            (policy_summary["corruption"] == corruption)
            & (policy_summary["severity_single"] == 0.4)
            & (policy_summary["variant"] == "uncal")
        ].copy()
    if corruption == "C3" and c3_eligible:
        sub = sub[sub["dataset"].astype(str).isin(c3_eligible)].copy()
    if sub.empty:
        return {}
    key_cols = ["dataset", "encoding", "model", "corruption"]
    base = sub[sub["policy"] == "fixed_0_5"][
        key_cols + ["f1_policy_mean", "expected_cost_policy_mean"]
    ].rename(
        columns={
            "f1_policy_mean": "f1_base",
            "expected_cost_policy_mean": "cost_base",
        }
    )
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for policy in ["fixed_0_5", "val_tuned_f1", "cost_sensitive"]:
        pol = sub[sub["policy"] == policy]
        merged = pol.merge(base, on=key_cols, how="inner")
        if merged.empty:
            continue
        out[policy] = {
            "delta_f1": (merged["f1_policy_mean"] - merged["f1_base"]).to_numpy(dtype=float),
            "delta_cost": (
                merged["expected_cost_policy_mean"] - merged["cost_base"]
            ).to_numpy(dtype=float),
            "threshold": merged["threshold_mean"].to_numpy(dtype=float),
        }
    return out


def build_policy_cost_sensitivity_table() -> None:
    policy_long = pd.read_csv(ROOT / "results" / "raw" / "policy_metrics_long.csv")
    policy_long["severity_single"] = pd.to_numeric(policy_long["severity"], errors="coerce")
    sev_json = policy_long["severity"].astype(str).str.startswith("{")
    if sev_json.any():
        parsed = policy_long.loc[sev_json, "severity"].apply(json.loads)
        policy_long.loc[sev_json, "severity_c2"] = parsed.apply(lambda x: x.get("C2"))
        policy_long.loc[sev_json, "severity_c4"] = parsed.apply(lambda x: x.get("C4"))

    policy_long = policy_long[policy_long["variant"] == "uncal"].copy()
    c3_eligible = _c3_eligible_datasets()
    ratios = [3.0, 5.0, 10.0]
    rows = []
    rng = np.random.default_rng(2060)

    scenarios = {
        "single_0.4": policy_long[
            (
                policy_long["corruption"].isin(["C2", "C4"])
                & (policy_long["severity_single"] == 0.4)
            )
            | (
                (policy_long["corruption"] == "C3")
                & (policy_long["severity_single"] == 0.4)
                & (policy_long["dataset"].astype(str).isin(c3_eligible))
            )
        ].copy(),
        "compound_0.4_0.4": policy_long[
            (policy_long["corruption"] == "C2+C4")
            & (policy_long["severity_c2"] == 0.4)
            & (policy_long["severity_c4"] == 0.4)
        ].copy(),
    }

    key_cols = ["dataset", "encoding", "model", "seed", "corruption", "severity"]
    for scenario, sub in scenarios.items():
        if sub.empty:
            continue
        base = sub[sub["policy"] == "fixed_0_5"][
            key_cols + ["tp", "tn", "fp", "fn"]
        ].rename(
            columns={
                "tp": "tp_base",
                "tn": "tn_base",
                "fp": "fp_base",
                "fn": "fn_base",
            }
        )
        if base.empty:
            continue

        for ratio in ratios:
            for policy, label in [
                ("fixed_0_5", "Fixed 0.5"),
                ("val_tuned_f1", "Validation-tuned (F1)"),
                ("cost_sensitive", "Cost-sensitive (t=1/6)"),
            ]:
                pol = sub[sub["policy"] == policy]
                if pol.empty:
                    continue
                merged = pol.merge(base, on=key_cols, how="inner")
                if merged.empty:
                    continue

                denom_pol = (
                    merged["tp"] + merged["tn"] + merged["fp"] + merged["fn"]
                ).clip(lower=1)
                denom_base = (
                    merged["tp_base"]
                    + merged["tn_base"]
                    + merged["fp_base"]
                    + merged["fn_base"]
                ).clip(lower=1)
                cost_pol = (merged["fp"] + ratio * merged["fn"]) / denom_pol
                cost_base = (merged["fp_base"] + ratio * merged["fn_base"]) / denom_base
                delta = (cost_pol - cost_base).to_numpy(dtype=float)
                ci_low, ci_high = _bootstrap_mean_ci(delta, seed=2061 + int(ratio * 10))

                rows.append(
                    {
                        "scenario": scenario,
                        "policy": policy,
                        "policy_label": label,
                        "cost_ratio_fn_to_fp": ratio,
                        "n_settings": int(len(delta)),
                        "expected_cost_mean": float(cost_pol.mean()),
                        "delta_cost_vs_fixed": float(delta.mean()),
                        "delta_cost_vs_fixed_ci95_low_boot": ci_low,
                        "delta_cost_vs_fixed_ci95_high_boot": ci_high,
                    }
                )

    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "table_policy_cost_sensitivity.csv", index=False)
    if table.empty:
        return

    lines = [
        r"\begin{tabularx}{\textwidth}{lrr>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X}",
        r"\toprule",
        r"\textbf{Scenario} & \textbf{$C_{FN}:C_{FP}$} & \textbf{Policy} & \textbf{Mean expected cost} & \textbf{Delta cost vs fixed (95\% CI)} \\",
        r"\midrule",
    ]
    scenario_label = {
        "single_0.4": "Single severe (C2/C3/C4; C3 eligible only)",
        "compound_0.4_0.4": "Compound severe (C2+C4)",
    }
    pol_order = {"fixed_0_5": 0, "val_tuned_f1": 1, "cost_sensitive": 2}
    table = table.sort_values(["scenario", "cost_ratio_fn_to_fp", "policy"], key=lambda col: col.map(pol_order) if col.name == "policy" else col)
    for _, row in table.iterrows():
        lines.append(
            f"{scenario_label.get(row['scenario'], row['scenario'])} & "
            f"{int(row['cost_ratio_fn_to_fp'])}:1 & {row['policy_label']} & "
            f"{_fmt(row['expected_cost_mean'], 4)} & "
            f"{_fmt_ci(row['delta_cost_vs_fixed'], row['delta_cost_vs_fixed_ci95_low_boot'], row['delta_cost_vs_fixed_ci95_high_boot'], 4, True)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    _write(TABLES / "table_policy_cost_sensitivity.tex", lines)


def build_policy_severe_table() -> None:
    policy_summary = pd.read_csv(TABLES / "policy_metrics_summary.csv")
    policy_summary["severity_single"] = pd.to_numeric(policy_summary["severity"], errors="coerce")
    c3_eligible = _c3_eligible_datasets()
    sev_json = policy_summary["severity"].astype(str).str.startswith("{")
    if sev_json.any():
        parsed = policy_summary.loc[sev_json, "severity"].apply(json.loads)
        policy_summary.loc[sev_json, "severity_c2"] = parsed.apply(lambda x: x.get("C2"))
        policy_summary.loc[sev_json, "severity_c4"] = parsed.apply(lambda x: x.get("C4"))
    def _scenario_subset(scenario: str) -> pd.DataFrame:
        if scenario == "single_0.4":
            mask_single = (
                policy_summary["corruption"].isin(["C2", "C4"])
                & (policy_summary["severity_single"] == 0.4)
            )
            mask_c3 = (
                (policy_summary["corruption"] == "C3")
                & (policy_summary["severity_single"] == 0.4)
            )
            if c3_eligible:
                mask_c3 = mask_c3 & policy_summary["dataset"].astype(str).isin(c3_eligible)
            sub = policy_summary[mask_single | mask_c3].copy()
        elif scenario == "compound_0.4_0.4":
            sub = policy_summary[
                (policy_summary["corruption"] == "C2+C4")
                & (policy_summary["severity_c2"] == 0.4)
                & (policy_summary["severity_c4"] == 0.4)
            ].copy()
        else:
            sub = pd.DataFrame()
        if sub.empty:
            return sub
        return sub[sub["variant"] == "uncal"].copy()

    def _policy_stats(sub: pd.DataFrame, policy: str, seed_base: int) -> Dict[str, float]:
        if sub.empty:
            return {
                "n_clusters": 0,
                "n_settings": 0,
                "threshold_mean": np.nan,
                "delta_f1_mean": np.nan,
                "delta_f1_ci95_low": np.nan,
                "delta_f1_ci95_high": np.nan,
                "delta_cost_mean": np.nan,
                "delta_cost_ci95_low": np.nan,
                "delta_cost_ci95_high": np.nan,
            }

        key_cols = ["dataset", "encoding", "model", "corruption"]
        base = sub[sub["policy"] == "fixed_0_5"][
            key_cols + ["f1_policy_mean", "expected_cost_policy_mean"]
        ].rename(
            columns={
                "f1_policy_mean": "f1_base",
                "expected_cost_policy_mean": "cost_base",
            }
        )
        pol = sub[sub["policy"] == policy]
        merged = pol.merge(base, on=key_cols, how="inner")
        if merged.empty:
            return {
                "n_clusters": 0,
                "n_settings": 0,
                "threshold_mean": np.nan,
                "delta_f1_mean": np.nan,
                "delta_f1_ci95_low": np.nan,
                "delta_f1_ci95_high": np.nan,
                "delta_cost_mean": np.nan,
                "delta_cost_ci95_low": np.nan,
                "delta_cost_ci95_high": np.nan,
            }

        merged = merged.copy()
        merged["delta_f1"] = merged["f1_policy_mean"] - merged["f1_base"]
        merged["delta_cost"] = merged["expected_cost_policy_mean"] - merged["cost_base"]

        ds = (
            merged.groupby("dataset", as_index=False)
            .agg(
                delta_f1=("delta_f1", "mean"),
                delta_cost=("delta_cost", "mean"),
                threshold_mean=("threshold_mean", "mean"),
            )
            .dropna()
        )
        delta_f1_vals = ds["delta_f1"].to_numpy(dtype=float)
        delta_cost_vals = ds["delta_cost"].to_numpy(dtype=float)

        f1_low, f1_high = _cluster_bootstrap_ci(delta_f1_vals, seed=seed_base + 1)
        cost_low, cost_high = _cluster_bootstrap_ci(delta_cost_vals, seed=seed_base + 2)

        return {
            "n_clusters": int(ds["dataset"].nunique()),
            "n_settings": int(len(merged)),
            "threshold_mean": float(ds["threshold_mean"].mean()),
            "delta_f1_mean": float(delta_f1_vals.mean()) if len(delta_f1_vals) else np.nan,
            "delta_f1_ci95_low": f1_low,
            "delta_f1_ci95_high": f1_high,
            "delta_cost_mean": float(delta_cost_vals.mean()) if len(delta_cost_vals) else np.nan,
            "delta_cost_ci95_low": cost_low,
            "delta_cost_ci95_high": cost_high,
        }

    single_sub = _scenario_subset("single_0.4")
    compound_sub = _scenario_subset("compound_0.4_0.4")

    rows = []
    for policy, label in [
        ("fixed_0_5", "Fixed 0.5"),
        ("val_tuned_f1", "Validation-tuned (F1)"),
        ("cost_sensitive", "Cost-sensitive"),
    ]:
        single_stats = _policy_stats(single_sub, policy=policy, seed_base=3040)
        comp_stats = _policy_stats(compound_sub, policy=policy, seed_base=4040)
        rows.append(
            {
                "label": label,
                "policy": policy,
                "n_single_clusters": single_stats["n_clusters"],
                "n_compound_clusters": comp_stats["n_clusters"],
                "n_single_settings": single_stats["n_settings"],
                "n_compound_settings": comp_stats["n_settings"],
                "thr": single_stats["threshold_mean"],
                "single_f1_mean": single_stats["delta_f1_mean"],
                "single_f1_low": single_stats["delta_f1_ci95_low"],
                "single_f1_high": single_stats["delta_f1_ci95_high"],
                "single_cost_mean": single_stats["delta_cost_mean"],
                "single_cost_low": single_stats["delta_cost_ci95_low"],
                "single_cost_high": single_stats["delta_cost_ci95_high"],
                "comp_f1_mean": comp_stats["delta_f1_mean"],
                "comp_f1_low": comp_stats["delta_f1_ci95_low"],
                "comp_f1_high": comp_stats["delta_f1_ci95_high"],
                "comp_cost_mean": comp_stats["delta_cost_mean"],
                "comp_cost_low": comp_stats["delta_cost_ci95_low"],
                "comp_cost_high": comp_stats["delta_cost_ci95_high"],
            }
        )

    severe_df = pd.DataFrame(rows)
    severe_df.rename(
        columns={
            "thr": "threshold_mean",
            "single_f1_mean": "delta_f1_single_mean",
            "single_f1_low": "delta_f1_single_ci95_low",
            "single_f1_high": "delta_f1_single_ci95_high",
            "single_cost_mean": "delta_cost_single_mean",
            "single_cost_low": "delta_cost_single_ci95_low",
            "single_cost_high": "delta_cost_single_ci95_high",
            "comp_f1_mean": "delta_f1_compound_mean",
            "comp_f1_low": "delta_f1_compound_ci95_low",
            "comp_f1_high": "delta_f1_compound_ci95_high",
            "comp_cost_mean": "delta_cost_compound_mean",
            "comp_cost_low": "delta_cost_compound_ci95_low",
            "comp_cost_high": "delta_cost_compound_ci95_high",
        }
    ).to_csv(TABLES / "table_threshold_policy_comparison_severe.csv", index=False)

    lines = [
        r"\begin{tabularx}{\textwidth}{lrrr>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X}",
        r"\toprule",
        r"\textbf{Policy} & \textbf{$n_{single,cl}$} & \textbf{$n_{compound,cl}$} & \textbf{Mean $t$} & \textbf{Delta F1 single (95\% CI)} & \textbf{Delta cost single (95\% CI)} & \textbf{Delta F1 C2+C4 (95\% CI)} & \textbf{Delta cost C2+C4 (95\% CI)} \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['label']} & {int(row['n_single_clusters'])} & {int(row['n_compound_clusters'])} & {_fmt(row['thr'], 4)} & "
            f"{_fmt_ci(row['single_f1_mean'], row['single_f1_low'], row['single_f1_high'], 4, True)} & "
            f"{_fmt_ci(row['single_cost_mean'], row['single_cost_low'], row['single_cost_high'], 4, True)} & "
            f"{_fmt_ci(row['comp_f1_mean'], row['comp_f1_low'], row['comp_f1_high'], 4, True)} & "
            f"{_fmt_ci(row['comp_cost_mean'], row['comp_cost_low'], row['comp_cost_high'], 4, True)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    _write(TABLES / "table_threshold_policy_comparison_severe.tex", lines)


def build_policy_pairwise_tests_table() -> None:
    path = TABLES / "table_threshold_policy_pairwise_tests.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return

    scenario_label = {
        "single_0.4": "Single severe",
        "compound_0.4_0.4": "Compound severe",
    }
    metric_label = {
        "delta_f1_vs_fixed": "Delta F1 (val-cost)",
        "delta_cost_vs_fixed": "Delta cost (val-cost)",
    }
    favored_label = {
        "val_tuned_f1": "validation-tuned",
        "cost_sensitive": "cost-sensitive",
        "tie": "tie",
    }
    lines = [
        r"\begin{tabularx}{\textwidth}{llrr>{\raggedright\arraybackslash}Xl>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X}",
        r"\toprule",
        r"\textbf{Scenario} & \textbf{Metric} & \textbf{$n$} & \textbf{$n_{\neq 0}$} & \textbf{Mean difference (95\% CI)} & \textbf{Favored} & \textbf{t p / q(BH)} & \textbf{W p / q(BH)} \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        delta_ci = _fmt_ci(
            row["delta_mean"],
            row["delta_ci95_low_boot"],
            row["delta_ci95_high_boot"],
            digits=4,
            signed=True,
        )
        t_cell = (
            f"{_fmt(row['paired_t_p_value'], 4)} / {_fmt(row['paired_t_p_value_bh'], 4)}"
            if pd.notna(row["paired_t_p_value"])
            else "-- / --"
        )
        w_cell = (
            f"{_fmt(row['wilcoxon_p_value'], 4)} / {_fmt(row['wilcoxon_p_value_bh'], 4)}"
            if pd.notna(row["wilcoxon_p_value"])
            else "-- / --"
        )
        lines.append(
            f"{scenario_label.get(str(row['scenario']), str(row['scenario']))} & "
            f"{metric_label.get(str(row['metric']), str(row['metric']))} & "
            f"{int(row['n_pairs_total'])} & {int(row['n_pairs_nonzero'])} & "
            f"{delta_ci} & {_latex_escape(favored_label.get(str(row['favored_policy_by_mean']), str(row['favored_policy_by_mean'])))} & {t_cell} & {w_cell} \\\\"
        )

    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    _write(TABLES / "table_threshold_policy_pairwise_tests.tex", lines)


def build_calibration_severe_table() -> None:
    calib_summary = pd.read_csv(TABLES / "calibration_metrics_summary.csv")
    calib_summary["severity_single"] = pd.to_numeric(calib_summary["severity"], errors="coerce")
    c3_eligible = _c3_eligible_datasets()
    sev_json = calib_summary["severity"].astype(str).str.startswith("{")
    if sev_json.any():
        parsed = calib_summary.loc[sev_json, "severity"].apply(json.loads)
        calib_summary.loc[sev_json, "severity_c2"] = parsed.apply(lambda x: x.get("C2"))
        calib_summary.loc[sev_json, "severity_c4"] = parsed.apply(lambda x: x.get("C4"))

    def _scenario_subset(scenario: str) -> pd.DataFrame:
        if scenario == "single_0.4":
            mask_single = (
                calib_summary["corruption"].isin(["C2", "C4"])
                & (calib_summary["severity_single"] == 0.4)
            )
            mask_c3 = (
                (calib_summary["corruption"] == "C3")
                & (calib_summary["severity_single"] == 0.4)
            )
            if c3_eligible:
                mask_c3 = mask_c3 & calib_summary["dataset"].astype(str).isin(c3_eligible)
            return calib_summary[mask_single | mask_c3].copy()
        if scenario == "compound_0.4_0.4":
            return calib_summary[
                (calib_summary["corruption"] == "C2+C4")
                & (calib_summary["severity_c2"] == 0.4)
                & (calib_summary["severity_c4"] == 0.4)
            ].copy()
        return pd.DataFrame()

    def _calibration_stats(sub: pd.DataFrame, calibrator: str, seed_base: int) -> Dict[str, float]:
        if sub.empty:
            return {
                "n_clusters": 0,
                "n_settings": 0,
                "delta_ece_mean": np.nan,
                "delta_ece_ci95_low": np.nan,
                "delta_ece_ci95_high": np.nan,
                "delta_logloss_mean": np.nan,
                "delta_logloss_ci95_low": np.nan,
                "delta_logloss_ci95_high": np.nan,
            }

        key_cols = ["dataset", "encoding", "model", "corruption"]
        base = sub[sub["calibrator"] == "uncal"][key_cols + ["ece_mean", "logloss_mean"]].rename(
            columns={"ece_mean": "ece_base", "logloss_mean": "logloss_base"}
        )
        csub = sub[sub["calibrator"] == calibrator][key_cols + ["ece_mean", "logloss_mean"]]
        merged = csub.merge(base, on=key_cols, how="inner")
        if merged.empty:
            return {
                "n_clusters": 0,
                "n_settings": 0,
                "delta_ece_mean": np.nan,
                "delta_ece_ci95_low": np.nan,
                "delta_ece_ci95_high": np.nan,
                "delta_logloss_mean": np.nan,
                "delta_logloss_ci95_low": np.nan,
                "delta_logloss_ci95_high": np.nan,
            }

        merged = merged.copy()
        merged["delta_ece"] = merged["ece_mean"] - merged["ece_base"]
        merged["delta_logloss"] = merged["logloss_mean"] - merged["logloss_base"]

        ds = (
            merged.groupby("dataset", as_index=False)
            .agg(
                delta_ece=("delta_ece", "mean"),
                delta_logloss=("delta_logloss", "mean"),
            )
            .dropna()
        )
        ece_vals = ds["delta_ece"].to_numpy(dtype=float)
        logloss_vals = ds["delta_logloss"].to_numpy(dtype=float)
        ece_low, ece_high = _cluster_bootstrap_ci(ece_vals, seed=seed_base + 1)
        ll_low, ll_high = _cluster_bootstrap_ci(logloss_vals, seed=seed_base + 2)

        return {
            "n_clusters": int(ds["dataset"].nunique()),
            "n_settings": int(len(merged)),
            "delta_ece_mean": float(ece_vals.mean()) if len(ece_vals) else np.nan,
            "delta_ece_ci95_low": ece_low,
            "delta_ece_ci95_high": ece_high,
            "delta_logloss_mean": float(logloss_vals.mean()) if len(logloss_vals) else np.nan,
            "delta_logloss_ci95_low": ll_low,
            "delta_logloss_ci95_high": ll_high,
        }

    single_sub = _scenario_subset("single_0.4")
    compound_sub = _scenario_subset("compound_0.4_0.4")

    rows = []
    lines = [
        r"\begin{tabularx}{\textwidth}{lrr>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X}",
        r"\toprule",
        r"\textbf{Calibrator} & \textbf{$n_{single,cl}$} & \textbf{$n_{compound,cl}$} & \textbf{Single severe: Delta ECE / Delta log loss (95\% CI)} & \textbf{Compound C2+C4: Delta ECE / Delta log loss (95\% CI)} \\",
        r"\midrule",
    ]
    labels = {
        "temp_scaled": "Temperature scaling",
        "platt": "Platt scaling",
        "isotonic": "Isotonic regression",
        "beta": "Beta calibration",
    }
    for calibrator in ["temp_scaled", "platt", "isotonic", "beta"]:
        s = _calibration_stats(single_sub, calibrator=calibrator, seed_base=5050)
        c = _calibration_stats(compound_sub, calibrator=calibrator, seed_base=6050)
        s_cell = (
            f"{_fmt_ci(s['delta_ece_mean'], s['delta_ece_ci95_low'], s['delta_ece_ci95_high'], 4, True)}; "
            f"{_fmt_ci(s['delta_logloss_mean'], s['delta_logloss_ci95_low'], s['delta_logloss_ci95_high'], 4, True)}"
        )
        c_cell = (
            f"{_fmt_ci(c['delta_ece_mean'], c['delta_ece_ci95_low'], c['delta_ece_ci95_high'], 4, True)}; "
            f"{_fmt_ci(c['delta_logloss_mean'], c['delta_logloss_ci95_low'], c['delta_logloss_ci95_high'], 4, True)}"
        )
        lines.append(
            f"{labels[calibrator]} & {int(s['n_clusters'])} & {int(c['n_clusters'])} & {s_cell} & {c_cell} \\\\"
        )
        rows.append(
            {
                "calibrator": calibrator,
                "calibrator_label": labels[calibrator],
                "n_single_clusters": int(s["n_clusters"]),
                "n_compound_clusters": int(c["n_clusters"]),
                "n_single_settings": int(s["n_settings"]),
                "n_compound_settings": int(c["n_settings"]),
                "delta_ece_single_mean": s["delta_ece_mean"],
                "delta_ece_single_ci95_low": s["delta_ece_ci95_low"],
                "delta_ece_single_ci95_high": s["delta_ece_ci95_high"],
                "delta_logloss_single_mean": s["delta_logloss_mean"],
                "delta_logloss_single_ci95_low": s["delta_logloss_ci95_low"],
                "delta_logloss_single_ci95_high": s["delta_logloss_ci95_high"],
                "delta_ece_compound_mean": c["delta_ece_mean"],
                "delta_ece_compound_ci95_low": c["delta_ece_ci95_low"],
                "delta_ece_compound_ci95_high": c["delta_ece_ci95_high"],
                "delta_logloss_compound_mean": c["delta_logloss_mean"],
                "delta_logloss_compound_ci95_low": c["delta_logloss_ci95_low"],
                "delta_logloss_compound_ci95_high": c["delta_logloss_ci95_high"],
            }
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    _write(TABLES / "table_calibration_comparison_severe.tex", lines)
    pd.DataFrame(rows).to_csv(TABLES / "table_calibration_comparison_severe.csv", index=False)


def build_calibration_pairwise_tests_table() -> None:
    path = TABLES / "table_calibration_pairwise_tests.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return

    scenario_label = {
        "single_0.4": "Single severe",
        "compound_0.4_0.4": "Compound severe",
    }
    metric_label = {
        "delta_ece_vs_uncal": "Delta ECE",
        "delta_logloss_vs_uncal": "Delta log loss",
    }
    cal_label = {
        "temp_scaled": "temperature",
        "platt": "platt",
        "isotonic": "isotonic",
        "beta": "beta",
        "tie": "tie",
    }
    lines = [
        r"\begin{tabularx}{\textwidth}{lllr>{\raggedright\arraybackslash}Xl>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X}",
        r"\toprule",
        r"\textbf{Scenario} & \textbf{Metric} & \textbf{A} & \textbf{B} & \textbf{A-B (95\% CI)} & \textbf{Favored} & \textbf{t p / q(BH)} & \textbf{W p / q(BH)} \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        delta_ci = _fmt_ci(
            row["delta_mean"],
            row["delta_ci95_low_boot"],
            row["delta_ci95_high_boot"],
            digits=4,
            signed=True,
        )
        t_cell = (
            f"{_fmt(row['paired_t_p_value'], 4)} / {_fmt(row['paired_t_p_value_bh'], 4)}"
            if pd.notna(row["paired_t_p_value"])
            else "-- / --"
        )
        w_cell = (
            f"{_fmt(row['wilcoxon_p_value'], 4)} / {_fmt(row['wilcoxon_p_value_bh'], 4)}"
            if pd.notna(row["wilcoxon_p_value"])
            else "-- / --"
        )
        lines.append(
            f"{scenario_label.get(str(row['scenario']), str(row['scenario']))} & "
            f"{metric_label.get(str(row['metric']), str(row['metric']))} & "
            f"{cal_label.get(str(row['calibrator_a']), str(row['calibrator_a']))} & "
            f"{cal_label.get(str(row['calibrator_b']), str(row['calibrator_b']))} & "
            f"{delta_ci} & {_latex_escape(cal_label.get(str(row['favored_calibrator_by_mean']), str(row['favored_calibrator_by_mean'])))} & {t_cell} & {w_cell} \\\\"
        )

    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    _write(TABLES / "table_calibration_pairwise_tests.tex", lines)


def build_c1_anchor_sensitivity_table() -> None:
    path = TABLES / "table_c1_anchor_sensitivity.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return

    mlabel = {"auc": "AUC", "f1": "F1", "ece": "ECE"}
    lines = [
        r"\begin{tabularx}{\textwidth}{l>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Label-informed Delta (95\% CI)} & \textbf{Label-agnostic Delta (95\% CI)} & \textbf{Agnostic - informed (95\% CI)} \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        informed = _fmt_ci(
            row["label_informed_mean"],
            row["label_informed_ci95_low_boot"],
            row["label_informed_ci95_high_boot"],
            digits=4,
            signed=True,
        )
        agnostic = _fmt_ci(
            row["label_agnostic_mean"],
            row["label_agnostic_ci95_low_boot"],
            row["label_agnostic_ci95_high_boot"],
            digits=4,
            signed=True,
        )
        gap = _fmt_ci(
            row["delta_gap_agnostic_minus_informed_mean"],
            row["delta_gap_agnostic_minus_informed_ci95_low_boot"],
            row["delta_gap_agnostic_minus_informed_ci95_high_boot"],
            digits=4,
            signed=True,
        )
        lines.append(f"{mlabel.get(str(row['metric']), str(row['metric']).upper())} & {informed} & {agnostic} & {gap} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    _write(TABLES / "table_c1_anchor_sensitivity.tex", lines)


def main() -> None:
    build_dataset_profiles()
    build_dataset_selection_audit()
    build_cross_dataset_effects()
    build_primary_inference()
    build_primary_dataset_inference()
    build_primary_hypothesis_tests()
    build_primary_hierarchical_sensitivity()
    build_policy_severe_table()
    build_policy_pairwise_tests_table()
    build_policy_cost_sensitivity_table()
    build_calibration_severe_table()
    build_calibration_pairwise_tests_table()
    build_c1_anchor_sensitivity_table()


if __name__ == "__main__":
    main()
