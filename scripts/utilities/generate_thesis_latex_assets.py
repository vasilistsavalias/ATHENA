"""Generate thesis-ready LaTeX tables/macros from the latest pipeline artifacts.

Usage:
    python scripts/utilities/generate_thesis_latex_assets.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(".")
OUTPUTS_13 = ROOT / "outputs" / "13_model_evaluation"
OUTPUTS_12 = ROOT / "outputs" / "12_model_training"
OUTPUTS_16 = ROOT / "outputs" / "16_expert_validation"
GENERATED = ROOT / "docs" / "thesis_latex" / "generated"


METRICS = ["psnr", "ssim", "lpips", "color", "pattern"]
PRIMARY_MODELS = [
    "FT-SD",
    "Vanilla SD",
    "Telea",
    "Navier-Stokes",
    "LaMa",
    "MAT",
    "CoModGAN",
    "FT-SD+TTA",
]


@dataclass(frozen=True)
class TableSpec:
    caption: str
    label: str


def _ensure_generated_dir() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)


def _fmt(x, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "NA"
    return f"{float(x):.{digits}f}"


def _fmt_p(p) -> str:
    if p is None or (isinstance(p, float) and pd.isna(p)):
        return "NA"
    p = float(p)
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _tex_escape_model(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).replace("_", r"\_") for c in out.columns]
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = (
                out[col]
                .astype(str)
                .str.replace("%", r"\%", regex=False)
                .str.replace("_", r"\_", regex=False)
            )
    if "model" in out.columns:
        out["model"] = out["model"].astype(str).str.replace("+", r"\+", regex=False)
    if "comparison" in out.columns:
        out["comparison"] = out["comparison"].astype(str).str.replace("+", r"\+", regex=False)
    return out


def _write_df_as_latex(df: pd.DataFrame, path: Path, spec: TableSpec, col_format: str | None = None) -> None:
    df = _tex_escape_model(df)
    latex = df.to_latex(
        index=False,
        escape=False,
        caption=spec.caption,
        label=spec.label,
        column_format=col_format,
    )
    path.write_text(latex, encoding="utf-8")


def _load_matrix_results() -> pd.DataFrame:
    full = OUTPUTS_13 / "benchmarking_matrix" / "matrix_results.full_test.csv"
    base = OUTPUTS_13 / "benchmarking_matrix" / "matrix_results.csv"
    path = full if full.exists() else base
    if not path.exists():
        raise FileNotFoundError(f"Matrix results file missing: {path}")
    return pd.read_csv(path)


def _build_unconditional_table(matrix_df: pd.DataFrame) -> pd.DataFrame:
    u = matrix_df[matrix_df["condition"] == "Unconditional"].copy()
    agg = (
        u.groupby("model", as_index=False)[METRICS]
        .mean()
        .rename(
            columns={
                "psnr": "PSNR (higher)",
                "ssim": "SSIM (higher)",
                "lpips": "LPIPS (lower)",
                "color": "COLOR (higher)",
                "pattern": "PATTERN (higher)",
            }
        )
    )
    agg = agg[agg["model"].isin(PRIMARY_MODELS)]
    agg["rank_psnr"] = agg["PSNR (higher)"].rank(ascending=False, method="min").astype(int)
    agg = agg.sort_values(["rank_psnr", "model"]).drop(columns=["rank_psnr"])
    for col in ["PSNR (higher)", "SSIM (higher)", "LPIPS (lower)", "COLOR (higher)", "PATTERN (higher)"]:
        agg[col] = agg[col].map(lambda x: float(f"{x:.4f}"))
    return agg


def _build_prompt_ablation_tables(matrix_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    def one(model_name: str) -> pd.DataFrame:
        d = matrix_df[matrix_df["model"] == model_name].copy()
        g = d.groupby("condition", as_index=False)[METRICS].mean()
        baseline = g[g["condition"] == "Unconditional"].iloc[0]
        g["Delta PSNR vs Uncond"] = g["psnr"] - float(baseline["psnr"])
        g["Delta LPIPS vs Uncond"] = g["lpips"] - float(baseline["lpips"])
        g = g.rename(
            columns={
                "condition": "Condition",
                "psnr": "PSNR (higher)",
                "ssim": "SSIM (higher)",
                "lpips": "LPIPS (lower)",
                "color": "COLOR (higher)",
                "pattern": "PATTERN (higher)",
            }
        )
        order = ["Unconditional", "Raw Text", "Enriched Text", "Refined Text (clip-safe)"]
        g["Condition"] = pd.Categorical(g["Condition"], categories=order, ordered=True)
        g = g.sort_values("Condition")
        for col in [
            "PSNR (higher)",
            "SSIM (higher)",
            "LPIPS (lower)",
            "COLOR (higher)",
            "PATTERN (higher)",
            "Delta PSNR vs Uncond",
            "Delta LPIPS vs Uncond",
        ]:
            g[col] = g[col].map(lambda x: float(f"{x:.4f}"))
        return g

    return one("FT-SD"), one("Vanilla SD")


def _build_fid_kid_table() -> pd.DataFrame:
    path = OUTPUTS_13 / "benchmarking_matrix" / "fid_kid_scores.csv"
    df = pd.read_csv(path)
    df = df[df["model"].isin(PRIMARY_MODELS)].copy()
    df = df.rename(columns={"FID": "FID (lower)", "KID_mean": "KID mean (lower)", "KID_std": "KID std"})
    df = df.sort_values("FID (lower)")
    for c in ["FID (lower)", "KID mean (lower)", "KID std"]:
        df[c] = df[c].map(lambda x: float(f"{x:.4f}"))
    return df


def _build_timing_table() -> pd.DataFrame:
    path = OUTPUTS_13 / "benchmarking_matrix" / "inference_timing.csv"
    df = pd.read_csv(path)
    df = df[df["model"].isin(PRIMARY_MODELS)].copy()
    df = df.rename(columns={"mean_ms": "Mean ms/image (lower)", "std_ms": "Std ms", "images_per_sec": "Images/sec (higher)"})
    df = df.sort_values("Mean ms/image (lower)")
    for c in ["Mean ms/image (lower)", "Std ms", "Images/sec (higher)"]:
        df[c] = df[c].map(lambda x: float(f"{x:.2f}"))
    return df


def _pick_comparison_row(
    pt: pd.DataFrame,
    *,
    model_a: str,
    condition_a: str,
    model_b: str,
    condition_b: str,
    metric: str,
) -> pd.Series | None:
    mask = (
        (pt["model_a"] == model_a)
        & (pt["condition_a"] == condition_a)
        & (pt["model_b"] == model_b)
        & (pt["condition_b"] == condition_b)
        & (pt["metric"] == metric)
    )
    if mask.any():
        return pt[mask].iloc[0]
    mask_rev = (
        (pt["model_a"] == model_b)
        & (pt["condition_a"] == condition_b)
        & (pt["model_b"] == model_a)
        & (pt["condition_b"] == condition_a)
        & (pt["metric"] == metric)
    )
    if mask_rev.any():
        row = pt[mask_rev].iloc[0].copy()
        row["mean_diff_a_minus_b"] = -float(row["mean_diff_a_minus_b"])
        row["t_statistic"] = -float(row["t_statistic"])
        row["cohens_d"] = -float(row["cohens_d"])
        return row
    return None


def _build_ftsd_significance_table() -> pd.DataFrame:
    path = OUTPUTS_13 / "statistical_tests" / "paired_t_tests.csv"
    pt = pd.read_csv(path)
    baselines = ["Telea", "Navier-Stokes", "Vanilla SD", "LaMa", "MAT", "CoModGAN"]

    rows: list[dict] = []
    for baseline in baselines:
        for metric in METRICS:
            row = _pick_comparison_row(
                pt,
                model_a="FT-SD",
                condition_a="Unconditional",
                model_b=baseline,
                condition_b="Unconditional",
                metric=metric,
            )
            if row is None:
                continue
            rows.append(
                {
                    "Comparison": f"FT-SD vs {baseline}",
                    "Metric": metric.upper(),
                    "Delta mean (FT-SD - baseline)": float(f"{float(row['mean_diff_a_minus_b']):.4f}"),
                    "p-value": _fmt_p(row["p_value"]),
                    "Bonferroni significant": "Yes" if bool(row["significant_bonferroni"]) else "No",
                    "Cohen's d": float(f"{float(row['cohens_d']):.3f}"),
                    "CI95 lower": float(f"{float(row['ci_95_lower']):.4f}"),
                    "CI95 upper": float(f"{float(row['ci_95_upper']):.4f}"),
                }
            )
    return pd.DataFrame(rows)


def _build_error_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    by_type = pd.read_csv(OUTPUTS_13 / "statistical_tests" / "error_by_mask_type.csv")
    by_sev = pd.read_csv(OUTPUTS_13 / "statistical_tests" / "error_by_severity.csv")
    keep_models = ["FT-SD", "Vanilla SD", "Telea", "LaMa", "MAT", "CoModGAN"]

    t = by_type[by_type["model"].isin(keep_models)].copy()
    t = t.rename(
        columns={
            "model": "model",
            "mask_type": "mask_type",
            "psnr_mean": "PSNR (higher)",
            "ssim_mean": "SSIM (higher)",
            "lpips_mean": "LPIPS (lower)",
            "color_mean": "COLOR (higher)",
            "pattern_mean": "PATTERN (higher)",
            "psnr_count": "N",
        }
    )
    t = t[["model", "mask_type", "N", "PSNR (higher)", "SSIM (higher)", "LPIPS (lower)", "COLOR (higher)", "PATTERN (higher)"]]
    for c in ["PSNR (higher)", "SSIM (higher)", "LPIPS (lower)", "COLOR (higher)", "PATTERN (higher)"]:
        t[c] = t[c].map(lambda x: float(f"{x:.4f}"))

    s = by_sev[by_sev["model"].isin(keep_models)].copy()
    s = s[s["severity_bin"] != "<10%"].copy()
    s = s.rename(
        columns={
            "model": "model",
            "severity_bin": "severity_bin",
            "psnr_mean": "PSNR (higher)",
            "ssim_mean": "SSIM (higher)",
            "lpips_mean": "LPIPS (lower)",
            "color_mean": "COLOR (higher)",
            "pattern_mean": "PATTERN (higher)",
            "psnr_count": "N",
        }
    )
    s = s[["model", "severity_bin", "N", "PSNR (higher)", "SSIM (higher)", "LPIPS (lower)", "COLOR (higher)", "PATTERN (higher)"]]
    for c in ["PSNR (higher)", "SSIM (higher)", "LPIPS (lower)", "COLOR (higher)", "PATTERN (higher)"]:
        s[c] = s[c].map(lambda x: float(f"{x:.4f}"))

    return t, s


def _build_macros(summary: dict) -> str:
    def macro(name: str, value: str) -> str:
        return f"\\newcommand{{\\{name}}}{{{value}}}"

    lines = [
        macro("AthenaDatasetEvalSamples", str(summary["evaluation"]["completed_samples"])),
        macro("AthenaModelsCompared", str(summary["evaluation"]["model_count"])),
        macro("AthenaBestPSNRModel", summary["rankings"]["psnr"]["model"].replace("+", r"\+")),
        macro("AthenaBestPSNRValue", _fmt(summary["rankings"]["psnr"]["value"], 4)),
        macro("AthenaBestLPIPSModel", summary["rankings"]["lpips"]["model"].replace("+", r"\+")),
        macro("AthenaBestLPIPSValue", _fmt(summary["rankings"]["lpips"]["value"], 4)),
        macro("AthenaSweepTrials", str(summary["training"]["sweep_trials"])),
        macro("AthenaFinalPassBestEpoch", str(summary["training"]["final_pass_best_epoch"])),
        macro("AthenaStageSixteenPairCount", str(summary["expert_pack"]["created_items"])),
    ]
    return "\n".join(lines) + "\n"


def _write_positioning_matrix_tex() -> pd.DataFrame:
    csv_path = ROOT / "docs" / "thesis" / "related_work_positioning_matrix.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df = df.copy()
    thesis_mask = df["Work"].str.contains("This thesis", case=False, na=False)
    if thesis_mask.any():
        idx = df[thesis_mask].index[0]
        df.loc[idx, "FID/KID + timing evaluation"] = "Yes"
        df.loc[idx, "Modern deep inpainting baselines (LaMa/MAT/etc.)"] = "Yes"
        df.loc[idx, "Expert validation study"] = "Stage16 pack + website ready"
        df.loc[idx, "Where worse / gaps"] = (
            "FT-SD does not dominate strongest deep baselines on all metrics; "
            "expert responses pending."
        )
    for col in df.columns:
        df[col] = df[col].fillna("Unknown")

    reduced = df[
        [
            "Work",
            "Year",
            "Domain",
            "Method family",
            "Text-Guided?" if "Text-Guided?" in df.columns else "Primary contribution",
        ]
    ].copy()
    _write_df_as_latex(
        reduced,
        GENERATED / "table_positioning_matrix_compact.tex",
        TableSpec(
            caption="Compact related-work positioning matrix (updated for current thesis status).",
            label="tab:positioning_compact",
        ),
    )
    df.to_csv(GENERATED / "positioning_matrix_updated.csv", index=False)
    return df


def main() -> int:
    _ensure_generated_dir()

    matrix_df = _load_matrix_results()
    eval_meta_path = OUTPUTS_13 / "benchmarking_matrix" / "evaluation_completeness.json"
    eval_meta = json.loads(eval_meta_path.read_text(encoding="utf-8")) if eval_meta_path.exists() else {}
    training_summary = json.loads((OUTPUTS_12 / "final_pass_summary.json").read_text(encoding="utf-8"))
    sweep_df = pd.read_csv(OUTPUTS_12 / "sweep" / "trials_summary.csv")
    stage16_summary = json.loads((OUTPUTS_16 / "stage_16_summary.json").read_text(encoding="utf-8"))

    unconditional = _build_unconditional_table(matrix_df)
    ftsd_prompt, vanilla_prompt = _build_prompt_ablation_tables(matrix_df)
    fid_kid = _build_fid_kid_table()
    timing = _build_timing_table()
    sig = _build_ftsd_significance_table()
    err_type, err_sev = _build_error_tables()

    _write_df_as_latex(
        unconditional,
        GENERATED / "table_unconditional_metrics.tex",
        TableSpec(
            caption="Unconditional full-test metrics across all evaluated models.",
            label="tab:unconditional_metrics",
        ),
    )
    _write_df_as_latex(
        ftsd_prompt,
        GENERATED / "table_prompt_ablation_ftsd.tex",
        TableSpec(
            caption="Prompt-conditioning ablation for FT-SD.",
            label="tab:prompt_ablation_ftsd",
        ),
    )
    _write_df_as_latex(
        vanilla_prompt,
        GENERATED / "table_prompt_ablation_vanilla.tex",
        TableSpec(
            caption="Prompt-conditioning ablation for Vanilla SD.",
            label="tab:prompt_ablation_vanilla",
        ),
    )
    _write_df_as_latex(
        fid_kid,
        GENERATED / "table_fid_kid.tex",
        TableSpec(
            caption="Distributional metrics (FID/KID). Lower is better.",
            label="tab:fid_kid",
        ),
    )
    _write_df_as_latex(
        timing,
        GENERATED / "table_inference_timing.tex",
        TableSpec(
            caption="Inference runtime comparison.",
            label="tab:inference_timing",
        ),
    )
    _write_df_as_latex(
        sig,
        GENERATED / "table_ftsd_vs_baselines_significance.tex",
        TableSpec(
            caption="Paired statistical tests for FT-SD versus major baselines (unconditional).",
            label="tab:ftsd_significance",
        ),
    )
    _write_df_as_latex(
        err_type,
        GENERATED / "table_error_by_mask_type.tex",
        TableSpec(
            caption="Error analysis by mask type (means over full-test set).",
            label="tab:error_mask_type",
        ),
    )
    _write_df_as_latex(
        err_sev,
        GENERATED / "table_error_by_severity.tex",
        TableSpec(
            caption="Error analysis by severity bin (means over full-test set).",
            label="tab:error_severity",
        ),
    )

    _write_positioning_matrix_tex()

    # Save source tables as CSV for easy inspection.
    unconditional.to_csv(GENERATED / "table_unconditional_metrics.csv", index=False)
    ftsd_prompt.to_csv(GENERATED / "table_prompt_ablation_ftsd.csv", index=False)
    vanilla_prompt.to_csv(GENERATED / "table_prompt_ablation_vanilla.csv", index=False)
    fid_kid.to_csv(GENERATED / "table_fid_kid.csv", index=False)
    timing.to_csv(GENERATED / "table_inference_timing.csv", index=False)
    sig.to_csv(GENERATED / "table_ftsd_vs_baselines_significance.csv", index=False)
    err_type.to_csv(GENERATED / "table_error_by_mask_type.csv", index=False)
    err_sev.to_csv(GENERATED / "table_error_by_severity.csv", index=False)

    summary = {
        "evaluation": {
            "phase": eval_meta.get("phase", "full_test"),
            "requested_samples": int(eval_meta.get("requested_samples", 0)),
            "completed_samples": int(eval_meta.get("completed_samples", 0)),
            "model_count": int(unconditional["model"].nunique()),
            "models": sorted(unconditional["model"].tolist()),
        },
        "rankings": {
            "psnr": {
                "model": unconditional.sort_values("PSNR (higher)", ascending=False).iloc[0]["model"],
                "value": float(unconditional.sort_values("PSNR (higher)", ascending=False).iloc[0]["PSNR (higher)"]),
            },
            "lpips": {
                "model": unconditional.sort_values("LPIPS (lower)", ascending=True).iloc[0]["model"],
                "value": float(unconditional.sort_values("LPIPS (lower)", ascending=True).iloc[0]["LPIPS (lower)"]),
            },
        },
        "training": {
            "sweep_trials": int(len(sweep_df)),
            "sweep_best_trial": str(sweep_df.sort_values("best_val_loss", ascending=True).iloc[0]["trial_id"]),
            "sweep_best_val_loss": float(sweep_df.sort_values("best_val_loss", ascending=True).iloc[0]["best_val_loss"]),
            "final_pass_best_epoch": int(training_summary.get("best_epoch", 0)),
            "final_pass_best_val_loss": float(training_summary.get("best_val_loss", 0.0)),
        },
        "expert_pack": {
            "created_items": int(stage16_summary.get("created_items", 0)),
            "method_pair": stage16_summary.get("method_pair", []),
        },
    }
    (GENERATED / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (GENERATED / "macros.tex").write_text(_build_macros(summary), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote generated thesis artifacts to: {GENERATED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
