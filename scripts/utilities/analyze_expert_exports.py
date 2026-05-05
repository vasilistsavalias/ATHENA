from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate analysis artifacts from ATHENA export bundle.")
    parser.add_argument("--responses-json", required=True)
    parser.add_argument("--responses-csv", required=True)
    parser.add_argument("--quality-report", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_counter(counter: Counter[str], title: str, output_path: Path) -> None:
    if plt is None:
        return
    labels = list(counter.keys()) or ["none"]
    values = [counter[label] for label in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel("Count")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _plot_response_times(rows: list[dict[str, Any]], output_path: Path) -> None:
    if plt is None:
        return
    by_block: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        block = str(row.get("block", "NA"))
        by_block[block].append(int(row.get("response_time_ms", 0)))
    labels = sorted(by_block.keys())
    means = [sum(by_block[label]) / max(1, len(by_block[label])) for label in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, means)
    ax.set_title("Mean response time by block")
    ax.set_ylabel("Milliseconds")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _write_csv(path: Path, rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    responses_json_path = Path(args.responses_json).resolve()
    quality_report_path = Path(args.quality_report).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    responses_json = _read_json(responses_json_path)
    quality_report = _read_json(quality_report_path)
    item_rows = list(responses_json.get("item_level", []))

    item_comment_total = len(item_rows)
    item_comment_non_empty = sum(1 for row in item_rows if str(row.get("comment", "")).strip())
    comment_coverage = (item_comment_non_empty / item_comment_total * 100.0) if item_comment_total else 0.0
    choice_counter = Counter(str(row.get("choice", "NA")) for row in item_rows if str(row.get("block", "")) == "B")
    confidence_counter = Counter(str(row.get("confidence", "NA")) for row in item_rows if str(row.get("block", "")) == "B")

    exceedance = quality_report.get("expert_plausibility_exceedance", {})
    _write_csv(
        output_dir / "exceedance_summary.csv",
        [
            ["cohort", "participant_count", "tier_2_met"],
            [
                "full_cohort",
                exceedance.get("full_cohort", {}).get("participant_count"),
                exceedance.get("full_cohort", {}).get("tier_2_met"),
            ],
            [
                "excluding_comprehension_risk",
                exceedance.get("excluding_comprehension_risk", {}).get("participant_count"),
                exceedance.get("excluding_comprehension_risk", {}).get("tier_2_met"),
            ],
        ],
    )

    (output_dir / "comment_completion.json").write_text(
        json.dumps(
            {
                "item_comment_total": item_comment_total,
                "item_comment_non_empty": item_comment_non_empty,
                "item_comment_coverage_percent": round(comment_coverage, 3),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_lines = [
        "# Production export analysis summary",
        "",
        f"- item comments (non-empty): {item_comment_non_empty}/{item_comment_total} ({comment_coverage:.1f}%)",
        f"- exceedance full_cohort tier_2_met: {exceedance.get('full_cohort', {}).get('tier_2_met')}",
        (
            "- exceedance excluding_comprehension_risk tier_2_met: "
            f"{exceedance.get('excluding_comprehension_risk', {}).get('tier_2_met')}"
        ),
        f"- findings_robust: {exceedance.get('findings_robust')}",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    _plot_counter(choice_counter, "Choice distribution (Block B)", output_dir / "choice_distribution.png")
    _plot_counter(confidence_counter, "Confidence distribution (Block B)", output_dir / "confidence_distribution.png")
    _plot_response_times(item_rows, output_dir / "response_time_summary.png")

    print(f"Analysis artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
