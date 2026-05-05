from __future__ import annotations

import argparse
from pathlib import Path

from app.db.session import SessionLocal
from app.services.ingest_service import import_pack


def parse_args():
    parser = argparse.ArgumentParser(
        description="Import a website-ready expert validation pack into the website database."
    )
    parser.add_argument("--pack-dir", required=True, help="Path to Expert_Pack_v2 directory.")
    parser.add_argument("--campaign-name", required=True, help="Campaign display name.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic campaign seed.")
    parser.add_argument(
        "--stage13-samples",
        default=None,
        help="Optional path to Stage 13/15 sample roots for Block A real images when importing two-block campaigns.",
    )
    parser.add_argument(
        "--no-activate",
        action="store_true",
        help="Import campaign without activating it.",
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Allow Block A and Block B to reuse the same underlying sample ids.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with SessionLocal() as db:
        campaign = import_pack(
            db,
            pack_dir=Path(args.pack_dir),
            campaign_name=args.campaign_name,
            seed=args.seed,
            stage13_samples=Path(args.stage13_samples) if args.stage13_samples else None,
            activate=not bool(args.no_activate),
            disjoint_blocks=not bool(args.allow_overlap),
        )
    print(
        f"Imported campaign id={campaign.id} name='{campaign.name}' "
        f"active={campaign.is_active} seed={campaign.seed}"
    )


if __name__ == "__main__":
    main()
