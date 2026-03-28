import json
import os
from typing import Iterable

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAPER_FIG_DPI = 300
PAPER_STYLE = {
    "font.size": 10.5,
    "axes.titlesize": 11.0,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.0,
    "figure.dpi": 160,
    "savefig.dpi": PAPER_FIG_DPI,
}
PAPER_REQUIRED_FIGS = [
    "paper__cross_dataset__severe_deltas.png",
    "paper__calibration__delta_vs_uncal.png",
    "paper__policy__summary.png",
]


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _set_plot_style():
    plt.rcParams.update(PAPER_STYLE)


def _clear_existing_pngs(path: str):
    for filename in os.listdir(path):
        if filename.endswith(".png"):
            try:
                os.remove(os.path.join(path, filename))
            except PermissionError:
                # Allow regeneration to proceed when a preview process keeps a file handle.
                continue


def _safe_savefig(fig, path: str, dpi: int = PAPER_FIG_DPI):
    try:
        fig.savefig(path, dpi=dpi)
    except PermissionError:
        # Keep the previous file if it is currently locked by another process.
        return


def _severity_to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_read_csv(path: str | None) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _iter_unique(df: pd.DataFrame, cols: Iterable[str]):
    grouped = df.groupby(list(cols))
    for key, _ in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        yield key


def plot_single_corruptions(summary_csv: str, plots_dir: str):
    df = pd.read_csv(summary_csv)
    _ensure_dir(plots_dir)
    metrics = ["auc", "f1", "ece"]
    corruptions = ["C1", "C2", "C3", "C4"]

    for dataset, encoding, model in _iter_unique(df, ["dataset", "encoding", "model"]):
        subset_base = df[
            (df["dataset"] == dataset)
            & (df["encoding"] == encoding)
            & (df["model"] == model)
        ]
        for corruption in corruptions:
            sub = subset_base[subset_base["corruption"] == corruption].copy()
            if sub.empty:
                continue
            sub["severity_num"] = _severity_to_float(sub["severity"])
            sub = sub.dropna(subset=["severity_num"])
            for metric in metrics:
                fig, ax = plt.subplots()
                for variant in ["uncal", "temp_scaled"]:
                    variant_sub = sub[sub["variant"] == variant]
                    if variant_sub.empty:
                        continue
                    variant_sub = variant_sub.sort_values("severity_num")
                    x = variant_sub["severity_num"].values
                    y = variant_sub[f"{metric}_mean"].values
                    ax.plot(
                        x,
                        y,
                        marker="o",
                        label=variant,
                    )
                    low_col = f"{metric}_ci95_low"
                    high_col = f"{metric}_ci95_high"
                    if low_col in variant_sub.columns and high_col in variant_sub.columns:
                        ax.fill_between(
                            x,
                            variant_sub[low_col].values,
                            variant_sub[high_col].values,
                            alpha=0.15,
                        )
                ax.set_title(f"{dataset} | {encoding} | {model} | {corruption} | {metric}")
                ax.set_xlabel("severity (alpha)")
                ax.set_ylabel(metric)
                ax.grid(alpha=0.25)
                ax.legend()
                fig.tight_layout()
                filename = f"{dataset}__{encoding}__{model}__{corruption}__{metric}.png"
                _safe_savefig(fig, os.path.join(plots_dir, filename), dpi=PAPER_FIG_DPI)
                plt.close(fig)


def plot_compound_heatmaps(summary_csv: str, plots_dir: str):
    df = pd.read_csv(summary_csv)
    _ensure_dir(plots_dir)
    metrics = ["auc", "ece"]

    sub = df[df["corruption"] == "C2+C4"].copy()
    if sub.empty:
        return

    def parse_severity(val):
        if isinstance(val, str) and val.startswith("{"):
            parsed = json.loads(val)
            return parsed.get("C2", 0.0), parsed.get("C4", 0.0)
        return (np.nan, np.nan)

    sub[["C2", "C4"]] = sub["severity"].apply(lambda x: pd.Series(parse_severity(x)))
    sub = sub.dropna(subset=["C2", "C4"])

    for dataset, encoding, model, variant in _iter_unique(
        sub, ["dataset", "encoding", "model", "variant"]
    ):
        base = sub[
            (sub["dataset"] == dataset)
            & (sub["encoding"] == encoding)
            & (sub["model"] == model)
            & (sub["variant"] == variant)
        ]
        for metric in metrics:
            pivot = base.pivot_table(
                index="C2",
                columns="C4",
                values=f"{metric}_mean",
                aggfunc="mean",
            )
            if pivot.empty:
                continue
            c2_vals = sorted(pivot.index.values)
            c4_vals = sorted(pivot.columns.values)
            data = pivot.loc[c2_vals, c4_vals].values

            fig, ax = plt.subplots()
            im = ax.imshow(data, origin="lower", aspect="auto")
            ax.set_xticks(range(len(c4_vals)), labels=[str(v) for v in c4_vals])
            ax.set_yticks(range(len(c2_vals)), labels=[str(v) for v in c2_vals])
            ax.set_xlabel("C4 severity")
            ax.set_ylabel("C2 severity")
            ax.set_title(f"{dataset} | {encoding} | {model} | {variant} | {metric}")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            filename = (
                f"{dataset}__{encoding}__{model}__C2plusC4__{variant}__{metric}_heatmap.png"
            )
            _safe_savefig(fig, os.path.join(plots_dir, filename), dpi=PAPER_FIG_DPI)
            plt.close(fig)


def plot_threshold_policy(policy_table_csv: str, plots_dir: str):
    if not os.path.exists(policy_table_csv):
        return

    df = pd.read_csv(policy_table_csv)
    if df.empty:
        return

    def _plot_delta_bars(sub: pd.DataFrame, delta_col: str, filename: str, title: str):
        if sub.empty:
            return
        policy_order = ["val_tuned_f1", "cost_sensitive"]
        corr_order = ["C2", "C3", "C4", "C2+C4"]
        sub = sub[sub["policy"].isin(policy_order)].copy()
        if sub.empty:
            return
        sub["policy"] = pd.Categorical(sub["policy"], categories=policy_order, ordered=True)
        sub["corruption"] = pd.Categorical(sub["corruption"], categories=corr_order, ordered=True)
        sub = sub.sort_values(["corruption", "policy"])
        labels = [f"{c}|{p}" for c, p in zip(sub["corruption"].astype(str), sub["policy"].astype(str))]
        vals = sub[delta_col].to_numpy()

        fig, ax = plt.subplots(figsize=(9, 4))
        colors = ["#1f77b4" if p == "val_tuned_f1" else "#2ca02c" for p in sub["policy"]]
        ax.bar(range(len(vals)), vals, color=colors, alpha=0.9)
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_xticks(range(len(vals)), labels=labels, rotation=35, ha="right")
        ax.set_ylabel(delta_col)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        _safe_savefig(fig, os.path.join(plots_dir, filename), dpi=PAPER_FIG_DPI)
        plt.close(fig)

    single_uncal = df[(df["scenario"] == "single_0.4") & (df["variant"] == "uncal")]
    comp_uncal = df[(df["scenario"] == "compound_0.4_0.4") & (df["variant"] == "uncal")]

    _plot_delta_bars(
        single_uncal,
        "delta_f1_vs_fixed",
        "policy__uncal__single0.4__delta_f1_vs_fixed.png",
        "Threshold policy effect on F1 (single corruptions, severity 0.4, uncal)",
    )
    _plot_delta_bars(
        single_uncal,
        "delta_cost_vs_fixed",
        "policy__uncal__single0.4__delta_cost_vs_fixed.png",
        "Threshold policy effect on expected cost (single corruptions, severity 0.4, uncal)",
    )
    _plot_delta_bars(
        comp_uncal,
        "delta_f1_vs_fixed",
        "policy__uncal__compound0.4_0.4__delta_f1_vs_fixed.png",
        "Threshold policy effect on F1 (compound C2+C4, uncal)",
    )
    _plot_delta_bars(
        comp_uncal,
        "delta_cost_vs_fixed",
        "policy__uncal__compound0.4_0.4__delta_cost_vs_fixed.png",
        "Threshold policy effect on expected cost (compound C2+C4, uncal)",
    )


def plot_paper_cross_dataset_effects(cross_table_csv: str, plots_dir: str):
    df = _safe_read_csv(cross_table_csv)
    if df.empty:
        return

    corruption_order = ["C1", "C2", "C3", "C4", "C2+C4 (0.4,0.4)"]
    metric_defs = [
        ("auc", "Delta AUC", "#1f77b4"),
        ("f1", "Delta F1", "#2ca02c"),
        ("ece", "Delta ECE", "#d62728"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=160)

    for ax, (metric, title, color) in zip(axes, metric_defs):
        sub = df[df["metric"] == metric].copy()
        if sub.empty:
            ax.set_axis_off()
            continue

        sub["corruption"] = pd.Categorical(
            sub["corruption"],
            categories=corruption_order,
            ordered=True,
        )
        sub = sub.sort_values("corruption")

        x = np.arange(len(sub))
        mean = sub["delta_mean_across_datasets"].to_numpy(dtype=float)
        low = sub["delta_ci95_low_boot"].to_numpy(dtype=float)
        high = sub["delta_ci95_high_boot"].to_numpy(dtype=float)
        yerr = np.vstack([mean - low, high - mean])

        ax.bar(x, mean, color=color, alpha=0.85)
        ax.errorbar(
            x,
            mean,
            yerr=yerr,
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=3,
        )
        ax.axhline(0.0, color="black", linewidth=1.0)
        labels = [
            str(c).replace(" (0.4,0.4)", "") for c in sub["corruption"].astype(str).tolist()
        ]
        ax.set_xticks(x, labels=labels, rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Severe - baseline effect")
    fig.suptitle(
        "Cross-dataset severe corruption effects with bootstrap 95% CIs (C3 uses categorical-eligible datasets only)"
    )
    fig.tight_layout()
    _safe_savefig(
        fig,
        os.path.join(plots_dir, "paper__cross_dataset__severe_deltas.png"),
        dpi=PAPER_FIG_DPI,
    )
    plt.close(fig)


def plot_paper_calibration_summary(calibration_table_csv: str, plots_dir: str):
    severe_csv = os.path.join(
        os.path.dirname(calibration_table_csv), "table_calibration_comparison_severe.csv"
    )
    severe = _safe_read_csv(severe_csv)
    if severe.empty:
        return

    calibrator_order = ["temp_scaled", "platt", "isotonic", "beta"]
    calib_labels = ["Temp", "Platt", "Isotonic", "Beta"]
    severe["calibrator"] = pd.Categorical(
        severe["calibrator"], categories=calibrator_order, ordered=True
    )
    severe = severe.sort_values("calibrator")
    merged = pd.DataFrame({"calibrator": calibrator_order}).merge(
        severe, on="calibrator", how="left"
    ).fillna(0.0)

    x = np.arange(len(calibrator_order))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=160)

    for ax, metric, title in [
        (axes[0], "ece", "Delta ECE vs uncal"),
        (axes[1], "logloss", "Delta log loss vs uncal"),
    ]:
        if metric == "ece":
            single_mean = merged["delta_ece_single_mean"].to_numpy(dtype=float)
            single_low = merged["delta_ece_single_ci95_low"].to_numpy(dtype=float)
            single_high = merged["delta_ece_single_ci95_high"].to_numpy(dtype=float)
            comp_mean = merged["delta_ece_compound_mean"].to_numpy(dtype=float)
            comp_low = merged["delta_ece_compound_ci95_low"].to_numpy(dtype=float)
            comp_high = merged["delta_ece_compound_ci95_high"].to_numpy(dtype=float)
        else:
            single_mean = merged["delta_logloss_single_mean"].to_numpy(dtype=float)
            single_low = merged["delta_logloss_single_ci95_low"].to_numpy(dtype=float)
            single_high = merged["delta_logloss_single_ci95_high"].to_numpy(dtype=float)
            comp_mean = merged["delta_logloss_compound_mean"].to_numpy(dtype=float)
            comp_low = merged["delta_logloss_compound_ci95_low"].to_numpy(dtype=float)
            comp_high = merged["delta_logloss_compound_ci95_high"].to_numpy(dtype=float)

        single_err = np.vstack([single_mean - single_low, single_high - single_mean])
        comp_err = np.vstack([comp_mean - comp_low, comp_high - comp_mean])
        ax.bar(
            x - width / 2,
            single_mean,
            width=width,
            color="#1f77b4",
            label="Single (C2/C3/C4 avg)",
            yerr=single_err,
            capsize=3,
        )
        ax.bar(
            x + width / 2,
            comp_mean,
            width=width,
            color="#ff7f0e",
            label="Compound C2+C4",
            yerr=comp_err,
            capsize=3,
        )
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_xticks(x, labels=calib_labels)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Change relative to uncalibrated")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Calibration gains under severe corruption (single severe: C2/C4 all datasets, C3 categorical-eligible only)"
    )
    fig.tight_layout()
    _safe_savefig(
        fig,
        os.path.join(plots_dir, "paper__calibration__delta_vs_uncal.png"),
        dpi=PAPER_FIG_DPI,
    )
    plt.close(fig)


def plot_paper_policy_summary(policy_table_csv: str, plots_dir: str):
    severe_csv = os.path.join(
        os.path.dirname(policy_table_csv), "table_threshold_policy_comparison_severe.csv"
    )
    df = _safe_read_csv(severe_csv)
    if df.empty:
        return

    policy_order = ["val_tuned_f1", "cost_sensitive"]
    policy_labels = ["Val-tuned F1", "Cost-sensitive"]
    df["policy"] = pd.Categorical(df["policy"], categories=["fixed_0_5"] + policy_order, ordered=True)
    sub = pd.DataFrame({"policy": policy_order}).merge(
        df[df["policy"].isin(policy_order)],
        on="policy",
        how="left",
    ).fillna(0.0)
    sub["cost_reduction_single_mean"] = -sub["delta_cost_single_mean"]
    sub["cost_reduction_single_ci95_low"] = -sub["delta_cost_single_ci95_high"]
    sub["cost_reduction_single_ci95_high"] = -sub["delta_cost_single_ci95_low"]
    sub["cost_reduction_compound_mean"] = -sub["delta_cost_compound_mean"]
    sub["cost_reduction_compound_ci95_low"] = -sub["delta_cost_compound_ci95_high"]
    sub["cost_reduction_compound_ci95_high"] = -sub["delta_cost_compound_ci95_low"]

    x = np.arange(len(policy_order))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=160)

    for ax, title, f1_col, f1_lo, f1_hi, c_col, c_lo, c_hi in [
        (
            axes[0],
            "Single severe (C2/C3/C4 avg)",
            "delta_f1_single_mean",
            "delta_f1_single_ci95_low",
            "delta_f1_single_ci95_high",
            "cost_reduction_single_mean",
            "cost_reduction_single_ci95_low",
            "cost_reduction_single_ci95_high",
        ),
        (
            axes[1],
            "Compound severe (C2+C4)",
            "delta_f1_compound_mean",
            "delta_f1_compound_ci95_low",
            "delta_f1_compound_ci95_high",
            "cost_reduction_compound_mean",
            "cost_reduction_compound_ci95_low",
            "cost_reduction_compound_ci95_high",
        ),
    ]:
        f1_mean = sub[f1_col].to_numpy(dtype=float)
        f1_low = sub[f1_lo].to_numpy(dtype=float)
        f1_high = sub[f1_hi].to_numpy(dtype=float)
        c_mean = sub[c_col].to_numpy(dtype=float)
        c_low = sub[c_lo].to_numpy(dtype=float)
        c_high = sub[c_hi].to_numpy(dtype=float)
        f1_err = np.vstack([f1_mean - f1_low, f1_high - f1_mean])
        c_err = np.vstack([c_mean - c_low, c_high - c_mean])
        ax.bar(
            x - width / 2,
            f1_mean,
            width=width,
            color="#2ca02c",
            label="Delta F1 vs fixed",
            yerr=f1_err,
            capsize=3,
        )
        ax.bar(
            x + width / 2,
            c_mean,
            width=width,
            color="#d62728",
            label="Cost reduction vs fixed",
            yerr=c_err,
            capsize=3,
        )
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_xticks(x, labels=policy_labels, rotation=12, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Improvement (+)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Policy governance gains under severe corruption (single severe: C2/C4 all datasets, C3 categorical-eligible only)"
    )
    fig.tight_layout()
    _safe_savefig(
        fig,
        os.path.join(plots_dir, "paper__policy__summary.png"),
        dpi=PAPER_FIG_DPI,
    )
    plt.close(fig)


def write_figure_qc_checklist(plots_dir: str):
    png_files = sorted([f for f in os.listdir(plots_dir) if f.lower().endswith(".png")])
    checklist_path = os.path.join(plots_dir, "figure_qc_checklist.txt")
    lines = []
    lines.append("Figure QC checklist (auto-generated)")
    lines.append(f"Configured save DPI: {PAPER_FIG_DPI}")
    lines.append(
        "Color semantics: harmful deltas use red tones for cost/ECE; improvement-friendly deltas use blue/green."
    )
    lines.append(f"Total PNG figures: {len(png_files)}")

    missing = [name for name in PAPER_REQUIRED_FIGS if name not in set(png_files)]
    if missing:
        lines.append("Required summary figures: FAIL")
        for name in missing:
            lines.append(f"  - missing: {name}")
    else:
        lines.append("Required summary figures: PASS")

    size_failures = []
    for name in PAPER_REQUIRED_FIGS:
        path = os.path.join(plots_dir, name)
        if not os.path.exists(path):
            continue
        img = mpimg.imread(path)
        h, w = int(img.shape[0]), int(img.shape[1])
        if w < 1400 or h < 450:
            size_failures.append((name, w, h))
        lines.append(f"Resolution {name}: {w}x{h}")

    if size_failures:
        lines.append("Summary figure readability (min 1400x450): FAIL")
        for name, w, h in size_failures:
            lines.append(f"  - {name}: {w}x{h}")
    else:
        lines.append("Summary figure readability (min 1400x450): PASS")

    small_files = []
    for name in png_files:
        path = os.path.join(plots_dir, name)
        size_bytes = os.path.getsize(path)
        if size_bytes < 20_000:
            small_files.append((name, size_bytes))
    if small_files:
        lines.append("Unexpectedly small PNG files (<20KB): WARN")
        for name, size_bytes in small_files[:20]:
            lines.append(f"  - {name}: {size_bytes} bytes")
    else:
        lines.append("Unexpectedly small PNG files (<20KB): PASS")

    with open(checklist_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_plot_index(plots_dir: str):
    rows = []
    for filename in sorted(os.listdir(plots_dir)):
        if not filename.endswith(".png"):
            continue
        stem = filename[:-4]
        parts = stem.split("__")
        if len(parts) == 5:
            dataset, encoding, model, corruption, metric = parts
            rows.append(
                {
                    "file": filename,
                    "dataset": dataset,
                    "encoding": encoding,
                    "model": model,
                    "corruption": corruption,
                    "variant": "",
                    "metric": metric,
                    "plot_type": "single",
                }
            )
        elif len(parts) == 6 and parts[3] == "C2plusC4":
            dataset, encoding, model, _, variant, metric_name = parts
            metric = metric_name.replace("_heatmap", "")
            rows.append(
                {
                    "file": filename,
                    "dataset": dataset,
                    "encoding": encoding,
                    "model": model,
                    "corruption": "C2+C4",
                    "variant": variant,
                    "metric": metric.replace("_heatmap", ""),
                    "plot_type": "compound_heatmap",
                }
            )
        elif len(parts) == 4 and parts[0] == "policy":
            _, variant, scenario, metric = parts
            rows.append(
                {
                    "file": filename,
                    "dataset": "",
                    "encoding": "",
                    "model": "",
                    "corruption": scenario,
                    "variant": variant,
                    "metric": metric,
                    "plot_type": "policy",
                }
            )
        elif len(parts) >= 3 and parts[0] == "paper":
            rows.append(
                {
                    "file": filename,
                    "dataset": "",
                    "encoding": "",
                    "model": "",
                    "corruption": parts[1],
                    "variant": "",
                    "metric": "__".join(parts[2:]),
                    "plot_type": "paper_summary",
                }
            )
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(plots_dir, "plot_index.csv"), index=False)


def generate_all_plots(
    summary_csv: str,
    plots_dir: str,
    policy_table_csv: str | None = None,
    cross_dataset_table_csv: str | None = None,
    calibration_table_csv: str | None = None,
):
    _ensure_dir(plots_dir)
    _set_plot_style()
    _clear_existing_pngs(plots_dir)
    plot_single_corruptions(summary_csv, plots_dir)
    plot_compound_heatmaps(summary_csv, plots_dir)
    policy_table_path = policy_table_csv or os.path.join(
        "results", "tables", "table_threshold_policy_comparison.csv"
    )
    cross_table_path = cross_dataset_table_csv or os.path.join(
        "results", "tables", "table_cross_dataset_effects.csv"
    )
    calibration_table_path = calibration_table_csv or os.path.join(
        "results", "tables", "table_calibration_comparison.csv"
    )
    if policy_table_path:
        plot_threshold_policy(policy_table_path, plots_dir)
    plot_paper_cross_dataset_effects(cross_table_path, plots_dir)
    plot_paper_calibration_summary(calibration_table_path, plots_dir)
    plot_paper_policy_summary(policy_table_path, plots_dir)
    write_figure_qc_checklist(plots_dir)
    write_plot_index(plots_dir)
