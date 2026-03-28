import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def _load_tables(root: Path) -> Dict[str, pd.DataFrame]:
    return {
        "summary": pd.read_csv(root / "results" / "metrics_summary.csv"),
        "single": pd.read_csv(root / "results" / "tables" / "table_single_corruption_delta.csv"),
        "cross": pd.read_csv(root / "results" / "tables" / "table_cross_dataset_effects.csv"),
        "compound": pd.read_csv(root / "results" / "tables" / "table_compound_delta.csv"),
        "effects": pd.read_csv(root / "results" / "tables" / "effects_seed_paired.csv"),
        "clean": pd.read_csv(root / "results" / "tables" / "table_clean_performance.csv"),
    }


def _compute_claims(tables: Dict[str, pd.DataFrame]) -> Dict:
    summary = tables["summary"]
    single = tables["single"]
    cross = tables["cross"]
    compound = tables["compound"]
    effects = tables["effects"]
    clean = tables["clean"]

    out: Dict = {}
    bh_col = "paired_t_p_value_bh_family_metric"
    if bh_col not in effects.columns:
        bh_col = "paired_t_p_value_bh"

    # Clean baseline: mean over models/encodings (uncal).
    clean_uncal = clean[clean["variant"] == "uncal"].copy()
    clean_agg = (
        clean_uncal.groupby("dataset")[["auc_mean", "f1_mean", "ece_mean"]]
        .mean()
        .reset_index()
        .sort_values("dataset")
    )
    out["clean_baseline_uncal"] = clean_agg.to_dict(orient="records")

    # Single-corruption cross-dataset means.
    single_means = cross[cross["corruption"].isin(["C1", "C2", "C3", "C4"])].copy()
    single_means = single_means.pivot_table(
        index="corruption",
        columns="metric",
        values="delta_mean_across_datasets",
        aggfunc="first",
    ).reset_index()
    single_means = single_means.rename(
        columns={
            "auc": "delta_auc_mean",
            "f1": "delta_f1_mean",
            "ece": "delta_ece_mean",
        }
    ).sort_values("corruption")
    out["single_cross_dataset_means"] = single_means.to_dict(orient="records")

    # Compound means.
    out["compound_cross_dataset_means"] = {
        "delta_auc_mean": float(compound["delta_auc_mean"].mean()),
        "delta_f1_mean": float(compound["delta_f1_mean"].mean()),
        "delta_ece_mean": float(compound["delta_ece_mean"].mean()),
    }

    # BH-significant counts.
    sig = effects[effects[bh_col] < 0.05]
    out["bh_significant_counts"] = (
        sig.groupby("metric").size().sort_index().to_dict()
    )

    # BH-significance rates by corruption and metric.
    rates = (
        effects.assign(sig=effects[bh_col] < 0.05)
        .groupby(["corruption", "metric"])["sig"]
        .mean()
        .mul(100.0)
        .unstack("metric")
        .fillna(0.0)
        .sort_index()
    )
    out["bh_significance_rates_percent"] = rates.reset_index().to_dict(orient="records")

    # Temperature scaling ECE delta.
    pivot = summary.pivot_table(
        index=["dataset", "encoding", "model", "corruption", "severity"],
        columns="variant",
        values="ece_mean",
    ).reset_index()
    pivot["delta"] = pivot["temp_scaled"] - pivot["uncal"]
    out["temp_scaling_ece_overall"] = {
        "mean_delta": float(pivot["delta"].mean()),
        "improved_count": int((pivot["delta"] < 0).sum()),
        "worsened_count": int((pivot["delta"] > 0).sum()),
        "total": int(len(pivot)),
    }

    sub = pivot[pivot["corruption"].isin(["C2", "C3", "C4"])].copy()
    sub["sev_num"] = pd.to_numeric(sub["severity"], errors="coerce")
    sub = sub[sub["sev_num"] == 0.4]
    per_dataset = []
    for dataset, ds in sub.groupby("dataset"):
        per_dataset.append(
            {
                "dataset": dataset,
                "mean_delta": float(ds["delta"].mean()),
                "improved_count": int((ds["delta"] < 0).sum()),
                "total": int(len(ds)),
            }
        )
    out["temp_scaling_ece_severity_0_4_c2_c3_c4"] = sorted(
        per_dataset, key=lambda x: x["dataset"]
    )

    # C3 encoding ablation at severity 0.4, uncal.
    c3 = summary[
        (summary["variant"] == "uncal") & (summary["corruption"] == "C3")
    ].copy()
    c3["sev_num"] = pd.to_numeric(c3["severity"], errors="coerce")
    c3 = c3[c3["sev_num"] == 0.4]
    left = c3[c3["encoding"] == "ignore_unknown"][
        ["dataset", "model", "auc_mean", "f1_mean", "ece_mean"]
    ]
    right = c3[c3["encoding"] == "unknown_bucket"][
        ["dataset", "model", "auc_mean", "f1_mean", "ece_mean"]
    ]
    merged = left.merge(right, on=["dataset", "model"], suffixes=("_ignore", "_unknown"))
    c3_diff = {}
    for metric in ["auc", "f1", "ece"]:
        d = merged[f"{metric}_mean_unknown"] - merged[f"{metric}_mean_ignore"]
        c3_diff[metric] = {
            "mean_diff_unknown_minus_ignore": float(d.mean()),
            "max_abs_diff": float(d.abs().max()),
        }
    out["c3_encoding_ablation_severity_0_4_uncal"] = c3_diff

    # Row counts.
    out["table_row_counts"] = {
        "summary_rows": int(len(summary)),
        "single_rows": int(len(single)),
        "cross_rows": int(len(cross)),
        "compound_rows": int(len(compound)),
        "effects_rows": int(len(effects)),
        "clean_rows": int(len(clean)),
    }

    return out


def _artifact_headlines(root: Path) -> Dict[str, float]:
    cross = pd.read_csv(root / "results" / "tables" / "table_cross_dataset_effects.csv")
    calib = pd.read_csv(root / "results" / "tables" / "table_calibration_comparison_severe.csv")
    policy = pd.read_csv(root / "results" / "tables" / "table_threshold_policy_comparison_severe.csv")

    out: Dict[str, float] = {}
    c2 = cross[cross["corruption"] == "C2"].set_index("metric")
    c2c4 = cross[cross["corruption"] == "C2+C4 (0.4,0.4)"].set_index("metric")
    out["c2_delta_auc"] = float(c2.loc["auc", "delta_mean_across_datasets"])
    out["c2_delta_f1"] = float(c2.loc["f1", "delta_mean_across_datasets"])
    out["c2_delta_ece"] = float(c2.loc["ece", "delta_mean_across_datasets"])
    out["c2c4_delta_auc"] = float(c2c4.loc["auc", "delta_mean_across_datasets"])
    out["c2c4_delta_f1"] = float(c2c4.loc["f1", "delta_mean_across_datasets"])

    calib_key = {
        "temp_scaled": "cal_temp_ece_single",
        "platt": "cal_platt_ece_single",
        "isotonic": "cal_isotonic_ece_single",
        "beta": "cal_beta_ece_single",
    }
    for cal, key in calib_key.items():
        row = calib[calib["calibrator"] == cal]
        if not row.empty:
            out[key] = float(row.iloc[0]["delta_ece_single_mean"])

    val = policy[policy["policy"] == "val_tuned_f1"]
    if not val.empty:
        out["policy_val_tuned_delta_f1_single"] = float(val.iloc[0]["delta_f1_single_mean"])
        out["policy_val_tuned_delta_cost_single"] = float(val.iloc[0]["delta_cost_single_mean"])
    cst = policy[policy["policy"] == "cost_sensitive"]
    if not cst.empty:
        out["policy_cost_sensitive_delta_cost_single"] = float(
            cst.iloc[0]["delta_cost_single_mean"]
        )
    return out


def _paper_headlines(paper_text: str) -> Dict[str, float]:
    patterns: Dict[str, Tuple[str, int]] = {
        "c2_delta_auc": (
            r"C2\s+(?:is|was|showed|shows|has)\s+(?:the\s+)?(?:dominant|strongest|largest)\s+(?:mean\s+)?single(?:-family)?\s+(?:stressor|severe\s+degradation)\s*\(Delta AUC ([+\-]?\d+\.\d+),\s*Delta F1 ([+\-]?\d+\.\d+),\s*Delta ECE ([+\-]?\d+\.\d+)\)",
            1,
        ),
        "c2_delta_f1": (
            r"C2\s+(?:is|was|showed|shows|has)\s+(?:the\s+)?(?:dominant|strongest|largest)\s+(?:mean\s+)?single(?:-family)?\s+(?:stressor|severe\s+degradation)\s*\(Delta AUC ([+\-]?\d+\.\d+),\s*Delta F1 ([+\-]?\d+\.\d+),\s*Delta ECE ([+\-]?\d+\.\d+)\)",
            2,
        ),
        "c2_delta_ece": (
            r"C2\s+(?:is|was|showed|shows|has)\s+(?:the\s+)?(?:dominant|strongest|largest)\s+(?:mean\s+)?single(?:-family)?\s+(?:stressor|severe\s+degradation)\s*\(Delta AUC ([+\-]?\d+\.\d+),\s*Delta F1 ([+\-]?\d+\.\d+),\s*Delta ECE ([+\-]?\d+\.\d+)\)",
            3,
        ),
        "c2c4_delta_auc": (
            r"(?:compound\s+)?C2\+C4\s+(?:amplifies degradation|caused the largest overall severe degradation|showed the largest mean severe degradation overall)\s*\(Delta AUC ([+\-]?\d+\.\d+),\s*Delta F1 ([+\-]?\d+\.\d+)\)",
            1,
        ),
        "c2c4_delta_f1": (
            r"(?:compound\s+)?C2\+C4\s+(?:amplifies degradation|caused the largest overall severe degradation|showed the largest mean severe degradation overall)\s*\(Delta AUC ([+\-]?\d+\.\d+),\s*Delta F1 ([+\-]?\d+\.\d+)\)",
            2,
        ),
        "policy_val_tuned_delta_f1_single": (
            r"validation-tuned thresholds improve F1 by ([+\-]?\d+\.\d+).*reduce expected cost by ([+\-]?\d+\.\d+).*Cost-sensitive thresholds reduce expected cost by ([+\-]?\d+\.\d+)",
            1,
        ),
        "policy_val_tuned_delta_cost_single": (
            r"validation-tuned thresholds improve F1 by ([+\-]?\d+\.\d+).*reduce expected cost by ([+\-]?\d+\.\d+).*Cost-sensitive thresholds reduce expected cost by ([+\-]?\d+\.\d+)",
            2,
        ),
        "policy_cost_sensitive_delta_cost_single": (
            r"validation-tuned thresholds improve F1 by ([+\-]?\d+\.\d+).*reduce expected cost by ([+\-]?\d+\.\d+).*Cost-sensitive thresholds reduce expected cost by ([+\-]?\d+\.\d+)",
            3,
        ),
        "cal_temp_ece_single": (
            r"Delta ECE versus uncalibrated is: temperature scaling ([+\-]?\d+\.\d+).*Platt ([+\-]?\d+\.\d+).*isotonic ([+\-]?\d+\.\d+).*beta ([+\-]?\d+\.\d+)",
            1,
        ),
        "cal_platt_ece_single": (
            r"Delta ECE versus uncalibrated is: temperature scaling ([+\-]?\d+\.\d+).*Platt ([+\-]?\d+\.\d+).*isotonic ([+\-]?\d+\.\d+).*beta ([+\-]?\d+\.\d+)",
            2,
        ),
        "cal_isotonic_ece_single": (
            r"Delta ECE versus uncalibrated is: temperature scaling ([+\-]?\d+\.\d+).*Platt ([+\-]?\d+\.\d+).*isotonic ([+\-]?\d+\.\d+).*beta ([+\-]?\d+\.\d+)",
            3,
        ),
        "cal_beta_ece_single": (
            r"Delta ECE versus uncalibrated is: temperature scaling ([+\-]?\d+\.\d+).*Platt ([+\-]?\d+\.\d+).*isotonic ([+\-]?\d+\.\d+).*beta ([+\-]?\d+\.\d+)",
            4,
        ),
    }

    out: Dict[str, float] = {}
    for key, (pat, idx) in patterns.items():
        m = re.search(pat, paper_text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            out[key] = float(m.group(idx))
    return out


def _verify_paper_numbers(root: Path, paper_path: Path, tolerance: float = 5e-4) -> Dict:
    artifact = _artifact_headlines(root)
    paper_text = paper_path.read_text(encoding="utf-8")
    paper_vals = _paper_headlines(paper_text)

    checks = []
    all_keys = sorted(set(artifact) | set(paper_vals))
    for key in all_keys:
        a = artifact.get(key)
        p = paper_vals.get(key)
        if a is None or p is None:
            checks.append(
                {
                    "key": key,
                    "status": "missing",
                    "artifact": a,
                    "paper": p,
                    "abs_diff": None,
                }
            )
            continue
        diff = abs(a - p)
        checks.append(
            {
                "key": key,
                "status": "ok" if diff <= tolerance else "mismatch",
                "artifact": float(a),
                "paper": float(p),
                "abs_diff": float(diff),
            }
        )

    n_ok = sum(1 for x in checks if x["status"] == "ok")
    n_total = len(checks)
    return {
        "paper_path": str(paper_path),
        "tolerance": float(tolerance),
        "n_ok": int(n_ok),
        "n_total": int(n_total),
        "all_pass": bool(n_ok == n_total),
        "checks": checks,
    }


def _print_text(claims: Dict):
    print("== Clean Baseline (uncal, mean over model+encoding) ==")
    for row in claims["clean_baseline_uncal"]:
        print(
            f"{row['dataset']}: "
            f"AUC={row['auc_mean']:.6f}, F1={row['f1_mean']:.6f}, ECE={row['ece_mean']:.6f}"
        )

    print("\n== Single-Corruption Cross-Dataset Means ==")
    for row in claims["single_cross_dataset_means"]:
        print(
            f"{row['corruption']}: "
            f"dAUC={row['delta_auc_mean']:.6f}, "
            f"dF1={row['delta_f1_mean']:.6f}, "
            f"dECE={row['delta_ece_mean']:.6f}"
        )

    c = claims["compound_cross_dataset_means"]
    print("\n== Compound Cross-Dataset Means ==")
    print(
        f"C2+C4(0.4,0.4): dAUC={c['delta_auc_mean']:.6f}, "
        f"dF1={c['delta_f1_mean']:.6f}, dECE={c['delta_ece_mean']:.6f}"
    )

    print("\n== BH Significant Counts ==")
    for metric, n in sorted(claims["bh_significant_counts"].items()):
        print(f"{metric}: {n}")

    t = claims["temp_scaling_ece_overall"]
    print("\n== Temp Scaling ECE Overall (temp - uncal) ==")
    print(
        f"mean_delta={t['mean_delta']:.9f}, improved={t['improved_count']}, "
        f"worsened={t['worsened_count']}, total={t['total']}"
    )

    print("\n== Temp Scaling ECE at severity=0.4 across C2/C3/C4 ==")
    for row in claims["temp_scaling_ece_severity_0_4_c2_c3_c4"]:
        print(
            f"{row['dataset']}: mean_delta={row['mean_delta']:.9f}, "
            f"improved={row['improved_count']}/{row['total']}"
        )

    print("\n== C3 Encoding Ablation (unknown - ignore, severity=0.4, uncal) ==")
    for metric, row in claims["c3_encoding_ablation_severity_0_4_uncal"].items():
        print(
            f"{metric}: mean_diff={row['mean_diff_unknown_minus_ignore']:.9f}, "
            f"max_abs={row['max_abs_diff']:.9f}"
        )

    print("\n== Row Counts ==")
    for k, v in claims["table_row_counts"].items():
        print(f"{k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="Extract manuscript headline numbers from results artifacts.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root containing results/ (default: current directory).",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output file. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--verify-paper",
        action="store_true",
        help="Validate manuscript headline numbers against CSV artifacts.",
    )
    parser.add_argument(
        "--paper-path",
        type=Path,
        default=Path("paper") / "paper.tex",
        help="Path to manuscript TeX file for --verify-paper mode.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5e-4,
        help="Absolute tolerance used by --verify-paper comparisons.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    tables = _load_tables(root)
    claims = _compute_claims(tables)
    verify_payload = None
    if args.verify_paper:
        verify_payload = _verify_paper_numbers(
            root=root,
            paper_path=(root / args.paper_path).resolve()
            if not args.paper_path.is_absolute()
            else args.paper_path.resolve(),
            tolerance=args.tolerance,
        )
        claims["paper_number_verification"] = verify_payload

    if args.format == "json":
        payload = json.dumps(claims, indent=2)
        if args.out:
            args.out.write_text(payload, encoding="utf-8")
        else:
            print(payload)
    else:
        if args.out:
            # For text output, write via temporary capture.
            from io import StringIO
            import sys

            old_stdout = sys.stdout
            buf = StringIO()
            sys.stdout = buf
            try:
                _print_text(claims)
            finally:
                sys.stdout = old_stdout
            args.out.write_text(buf.getvalue(), encoding="utf-8")
        else:
            _print_text(claims)

    if verify_payload is not None:
        if not verify_payload["all_pass"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
