import argparse
import json
import os
import sys
from pathlib import Path


def _bootstrap_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def main() -> int:
    _bootstrap_imports()

    from thesis_pipeline.components.data_acquisition_europeana import EuropeanaOpenAcquirer
    from thesis_pipeline.utils.env_loader import load_env_file

    parser = argparse.ArgumentParser(description="Quick Europeana downloader smoke test.")
    parser.add_argument("--limit", type=int, default=10, help="Number of images to download.")
    parser.add_argument("--query", default="ancient greek vase", help="Europeana search query.")
    parser.add_argument("--rows", type=int, default=50, help="Rows per API page.")
    parser.add_argument(
        "--out",
        default="smoke_tests/europeana_only",
        help="Output directory for test assets.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    load_env_file(repo_root / ".env")

    output_root = Path(args.out)
    images_dir = output_root / "images"
    metadata_dir = output_root / "metadata"
    state_file = output_root / "europeana_state.json"
    output_root.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("EUROPEANA_API_KEY", "").strip()
    if not api_key:
        print("ERROR: EUROPEANA_API_KEY not found in env/.env")
        return 1

    acquirer = EuropeanaOpenAcquirer(api_key=api_key)
    summary = acquirer.download(
        query=args.query,
        output_dir=images_dir,
        metadata_dir=metadata_dir,
        limit=max(1, int(args.limit)),
        rows=max(1, int(args.rows)),
        reusability="open",
        state_file=state_file,
    )

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary.__dict__, indent=2), encoding="utf-8")
    print(json.dumps(summary.__dict__, indent=2))
    print(f"\nSaved: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
