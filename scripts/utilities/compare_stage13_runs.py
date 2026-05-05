from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _stage13_paths(run_root: Path) -> dict[str, Path]:
    stage13 = run_root / "13_model_evaluation"
    return {
        "stage13": stage13,
        "matrix": stage13 / "benchmarking_matrix" / "matrix_results.csv",
        "fid_kid": stage13 / "benchmarking_matrix" / "fid_kid_scores.csv",
        "timing": stage13 / "benchmarking_matrix" / "inference_timing.csv",
        "paired": stage13 / "statistical_tests" / "paired_t_tests.csv",
    }


def _best_by_metric(df: pd.DataFrame, metric: str, higher_is_better: bool) -> tuple[str, float] | None:
    if df is None or df.empty or metric not in df.columns:
        return None
    # Only compare unconditional rows for a clean apples-to-apples view.
    if "condition" in df.columns:
        df = df[df["condition"] == "Unconditional"]
    g = df.groupby("model")[metric].mean()
    if g.empty:
        return None
    best = g.idxmax() if higher_is_better else g.idxmin()
    return str(best), float(g.loc[best])


def _markdown_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_(missing or empty)_"
    return df.to_markdown(index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two pipeline runs (Stage 13 artifacts).")
    parser.add_argument("--run-a", required=True, type=Path, help="Run A outputs root (contains 13_model_evaluation/).")
    parser.add_argument("--run-b", required=True, type=Path, help="Run B outputs root (contains 13_model_evaluation/).")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for comparison report.")
    args = parser.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    a = _stage13_paths(args.run_a)
    b = _stage13_paths(args.run_b)

    a_matrix = _read_csv(a["matrix"])
    b_matrix = _read_csv(b["matrix"])
    a_fid = _read_csv(a["fid_kid"])
    b_fid = _read_csv(b["fid_kid"])
    a_timing = _read_csv(a["timing"])
    b_timing = _read_csv(b["timing"])

    # Best-per-metric summary
    metrics = [
        ("psnr", True),
        ("ssim", True),
        ("lpips", False),
        ("color", True),
        ("pattern", True),
    ]
    summary_rows = []
    for metric, hib in metrics:
        a_best = _best_by_metric(a_matrix, metric, hib)
        b_best = _best_by_metric(b_matrix, metric, hib)
        summary_rows.append(
            {
                "metric": metric,
                "higher_is_better": hib,
                "run_a_best_model": a_best[0] if a_best else None,
                "run_a_best_value": round(a_best[1], 4) if a_best else None,
                "run_b_best_model": b_best[0] if b_best else None,
                "run_b_best_value": round(b_best[1], 4) if b_best else None,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "stage13_metric_winners.csv", index=False)

    # FID/KID + Timing merge (if present)
    def _prep_fid(df: pd.DataFrame | None, label: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["model", f"FID_{label}", f"KID_mean_{label}", f"KID_std_{label}"])
        out = df.copy()
        out = out.rename(
            columns={
                "FID": f"FID_{label}",
                "KID_mean": f"KID_mean_{label}",
                "KID_std": f"KID_std_{label}",
            }
        )
        return out[["model", f"FID_{label}", f"KID_mean_{label}", f"KID_std_{label}"]]

    def _prep_timing(df: pd.DataFrame | None, label: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["model", f"mean_ms_{label}"])
        out = df.copy()
        # expected columns: model, mean_ms
        if "mean_ms" in out.columns:
            out = out.rename(columns={"mean_ms": f"mean_ms_{label}"})
            return out[["model", f"mean_ms_{label}"]]
        return pd.DataFrame(columns=["model", f"mean_ms_{label}"])

    fid_merge = _prep_fid(a_fid, "a").merge(_prep_fid(b_fid, "b"), on="model", how="outer")
    timing_merge = _prep_timing(a_timing, "a").merge(_prep_timing(b_timing, "b"), on="model", how="outer")

    fid_merge.to_csv(out_dir / "fid_kid_comparison.csv", index=False)
    timing_merge.to_csv(out_dir / "timing_comparison.csv", index=False)

    report = []
    report.append("# Stage 13 Run Comparison\n")
    report.append(f"- Run A: `{args.run_a}`")
    report.append(f"- Run B: `{args.run_b}`\n")

    report.append("## Metric winners (Unconditional means)\n")
    report.append(_markdown_table(summary_df))
    report.append("\n\n## FID/KID comparison (if available)\n")
    report.append(_markdown_table(fid_merge))
    report.append("\n\n## Inference timing comparison (if available)\n")
    report.append(_markdown_table(timing_merge))
    report.append("\n")

    (out_dir / "stage13_run_comparison.md").write_text("\n".join(report), encoding="utf-8")

    # Lightweight JSON for programmatic use
    payload = {
        "run_a": str(args.run_a),
        "run_b": str(args.run_b),
        "metric_winners": summary_rows,
    }
    (out_dir / "stage13_run_comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

