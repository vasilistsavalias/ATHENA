import argparse
import csv
import json
from pathlib import Path


def _to_float(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.upper() == "N/A":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _percentile(sorted_values, p):
    if not sorted_values:
        return None
    if p <= 0:
        return float(sorted_values[0])
    if p >= 1:
        return float(sorted_values[-1])
    idx = int(round(p * (len(sorted_values) - 1)))
    return float(sorted_values[idx])


def summarize_focus_scores(rejection_log_csv: Path):
    """
    Reads Stage 03 rejection_log.csv (new schema) and summarizes focus_score distributions.

    Only crop-level rows are used (crop_index != N/A).
    """
    accepted_scores = []
    rejected_scores = []
    accepted_reasons = {}
    rejected_reasons = {}

    with open(rejection_log_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            crop_index = (row.get("crop_index") or "").strip()
            if crop_index in ("", "N/A"):
                continue

            status = (row.get("status") or "").strip()
            reason = (row.get("reason") or "").strip()
            score = _to_float(row.get("focus_score"))

            if score is None:
                continue

            if status == "Accepted":
                accepted_scores.append(score)
                accepted_reasons[reason] = accepted_reasons.get(reason, 0) + 1
            else:
                rejected_scores.append(score)
                rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1

    accepted_scores.sort()
    rejected_scores.sort()

    def dist(values):
        return {
            "n": len(values),
            "min": _percentile(values, 0.0),
            "p10": _percentile(values, 0.10),
            "p25": _percentile(values, 0.25),
            "median": _percentile(values, 0.50),
            "p75": _percentile(values, 0.75),
            "p90": _percentile(values, 0.90),
            "max": _percentile(values, 1.0),
        }

    return {
        "accepted_focus_score": dist(accepted_scores),
        "rejected_focus_score": dist(rejected_scores),
        "accepted_reasons": accepted_reasons,
        "rejected_reasons": rejected_reasons,
    }


def suggest_thresholds(summary):
    """
    Heuristic suggestions:
    - main threshold: p10 of accepted (keeps ~90% of currently-accepted crops).
    - relaxed threshold: p05 of accepted (keeps ~95% with fallback conditions).
    If accepted distribution missing, return None suggestions.
    """
    acc = summary.get("accepted_focus_score", {}) or {}
    if not acc.get("n"):
        return {"blur_threshold_main": None, "blur_threshold_relaxed": None}

    main = acc.get("p10")
    relaxed = acc.get("p25")  # slightly more permissive than main gating when combined with conf+area fallback
    return {
        "blur_threshold_main": main,
        "blur_threshold_relaxed": relaxed,
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate Stage 03 focus-score thresholds from rejection_log.csv.")
    parser.add_argument(
        "--rejection-log",
        type=str,
        default="outputs/03_intelligent_filtering/rejection_log.csv",
        help="Path to Stage 03 rejection_log.csv (new schema).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/03_intelligent_filtering/calibration_report.json",
        help="Path to write calibration JSON report.",
    )
    parser.add_argument(
        "--suggest",
        action="store_true",
        help="Include threshold suggestions in the report.",
    )
    args = parser.parse_args()

    rejection_log_csv = Path(args.rejection_log)
    if not rejection_log_csv.exists():
        raise SystemExit(f"Missing file: {rejection_log_csv}")

    summary = summarize_focus_scores(rejection_log_csv)
    report = {"source": str(rejection_log_csv), "summary": summary}
    if args.suggest:
        report["suggested_thresholds"] = suggest_thresholds(summary)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=4), encoding="utf-8")
    print(f"Wrote calibration report: {out_path}")


if __name__ == "__main__":
    main()

