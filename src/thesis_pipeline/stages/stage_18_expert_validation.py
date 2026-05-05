import json
import logging
import random
import re
import shutil
from datetime import datetime, timezone
from hashlib import md5
from pathlib import Path

import pandas as pd

from thesis_pipeline.config_manager import ConfigManager


class ExpertValidationStage:
    """Stage 18 — Expert validation pack.

    Supports two modes:
    - ``"top1_vs_real"``: compare the composite-ranked top-1 method's
      restorations against real undamaged ground-truth images in a blinded
      A/B survey.  The expert task is "which image looks more natural?"
    - ``"method_pair"`` (legacy): compare two named restoration methods.

    Design goals:
    - No ground truth label shown (avoid bias).
    - Deterministic sampling (seeded) with basic stratification by mask type/coverage.
    - Website/tool friendly manifests.
    """

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.paths = config_manager.get_paths()
        self.config = config_manager.config
        self.logger = logging.getLogger(__name__)
        self.output_dir = self.config_manager.get_stage_artifact_dir("S18")

        try:
            self.seed = int(self.config.global_params.get("random_state", 42))
        except Exception:
            self.seed = 42

        cfg = self.config.get("expert_validation", {}) if self.config else {}
        self.samples_to_select = int(getattr(cfg, "samples_to_select", 20) or 20)
        self.condition = str(getattr(cfg, "condition", "Unconditional") or "Unconditional")
        self.practice_samples = int(getattr(cfg, "practice_samples", 3) or 3)

        # Mode selection: top1_vs_real (V7) or method_pair (legacy)
        self.mode = str(getattr(cfg, "mode", "top1_vs_real") or "top1_vs_real")
        # Legacy fallback
        self.method_pair = list(getattr(cfg, "method_pair", ["Telea", "FT-SD"]) or ["Telea", "FT-SD"])

        self.eval_root = self.config_manager.get_stage_artifact_dir("S15")
        self.samples_dir = self.eval_root / "samples"
        self.matrix_csv = self.eval_root / "benchmarking_matrix" / "matrix_results.csv"
        self.test_gt_dir = Path(self.paths.data.inpainting) / "test" / "ground_truth"

        # Ground-truth directory (for top1_vs_real)
        self.gt_dir = self.config_manager.get_stage_artifact_path("S02", "ground_truth")

    @staticmethod
    def _slug(x: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", str(x)).strip("_")

    def _resolve_sample_dir(self, sample_id: str) -> Path | None:
        candidates = [
            self.samples_dir / str(sample_id),
            self.samples_dir / Path(str(sample_id)).stem,
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate
        return None

    def _select_samples(self) -> list[str]:
        """Deterministic, lightly stratified selection using Stage 15 matrix_results.csv."""
        rng = random.Random(self.seed)

        # Fallback: list sample folders if matrix CSV missing.
        if not self.matrix_csv.exists():
            sample_ids = sorted([d.name for d in self.samples_dir.iterdir() if d.is_dir()])
            rng.shuffle(sample_ids)
            return sample_ids[: min(len(sample_ids), self.samples_to_select)]

        df = pd.read_csv(self.matrix_csv)
        if df.empty:
            return []

        # Use one row per sample_id (mask coverage/type are identical across models).
        df = df[df["condition"] == "Unconditional"].copy()
        df = df.groupby("sample_id", as_index=False)[["mask_type", "mask_coverage"]].first()

        if "mask_coverage" in df.columns:
            df["coverage_bin"] = pd.cut(
                df["mask_coverage"],
                bins=[0.0, 0.10, 0.25, 0.50, 1.0],
                labels=["<10%", "10-25%", "25-50%", ">50%"],
                include_lowest=True,
            ).astype(str)
        else:
            df["coverage_bin"] = "unknown"

        selected: list[str] = []
        remaining = int(self.samples_to_select)

        # First pass: pick 1 from each non-empty (mask_type, coverage_bin) group.
        groups = list(df.groupby(["mask_type", "coverage_bin"]))
        # Stable ordering for determinism
        groups.sort(key=lambda g: (str(g[0][0]), str(g[0][1])))
        for (_, _), gdf in groups:
            if remaining <= 0:
                break
            candidates = sorted(gdf["sample_id"].tolist())
            rng.shuffle(candidates)
            if candidates:
                selected.append(candidates[0])
                remaining -= 1

        # Second pass: fill remaining from all other candidates.
        if remaining > 0:
            pool = sorted(list(set(df["sample_id"].tolist()) - set(selected)))
            rng.shuffle(pool)
            selected.extend(pool[:remaining])

        return selected

    # ------------------------------------------------------------------
    # Top-1 helpers
    # ------------------------------------------------------------------
    def _load_top1_method(self) -> str:
        """Read composite ranking's ``top1_method.json`` produced by S15."""
        top1_json = self.eval_root / "benchmarking_matrix" / "top1_method.json"
        if not top1_json.exists():
            raise FileNotFoundError(
                f"top1_method.json not found at {top1_json}. "
                "Run S15 (model evaluation) with composite ranking first."
            )
        payload = json.loads(top1_json.read_text(encoding="utf-8"))
        method = payload.get("top1_method") or payload.get("method") or payload.get("model")
        if not method:
            raise ValueError(
                "top1_method.json is missing a method key "
                f"('top1_method', 'method', or 'model'): {payload}"
            )
        return str(method)

    def _find_ground_truth(self, sample_id: str) -> Path | None:
        """Locate the real undamaged ground-truth image for *sample_id*.

        Searches the current evaluation sample directory first (Stage 15 saves
        ``original.png`` there), then the canonical test split ground-truth
        directory, and finally older acquisition-era locations.
        """
        stem = Path(sample_id).stem
        sample_dir = self._resolve_sample_dir(sample_id)
        parents: list[Path] = []
        if sample_dir is not None:
            parents.append(sample_dir)
        parents.extend([self.test_gt_dir, self.gt_dir])

        for parent in parents:
            if not parent.exists():
                continue
            original_png = parent / "original.png"
            if original_png.exists():
                return original_png
            for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                for candidate_name in [
                    f"ground_truth{suffix}",
                    f"gt{suffix}",
                    str(sample_id),
                    f"{stem}{suffix}",
                    f"{stem}_gt{suffix}",
                    f"{stem}_original{suffix}",
                ]:
                    p = parent / candidate_name
                    if p.exists():
                        return p
        return None

    # ------------------------------------------------------------------
    # Metadata helper
    # ------------------------------------------------------------------
    def _build_meta_lookup(self) -> dict[str, dict]:
        meta_lookup: dict[str, dict] = {}
        try:
            if self.matrix_csv.exists():
                dfm = pd.read_csv(self.matrix_csv)
                dfm = dfm[dfm["condition"] == "Unconditional"].copy()
                dfm = dfm.groupby("sample_id", as_index=False)[["mask_type", "mask_coverage"]].first()
                dfm["coverage_bin"] = pd.cut(
                    dfm["mask_coverage"],
                    bins=[0.0, 0.10, 0.25, 0.50, 1.0],
                    labels=["<10%", "10-25%", "25-50%", ">50%"],
                    include_lowest=True,
                ).astype(str)
                meta_lookup = {
                    r["sample_id"]: {
                        "mask_type": r.get("mask_type", "unknown"),
                        "mask_coverage": float(r.get("mask_coverage", 0.0)),
                        "coverage_bin": r.get("coverage_bin", "unknown"),
                    }
                    for r in dfm.to_dict(orient="records")
                }
        except Exception:
            pass
        return meta_lookup

    def _anchor_sample_ids(self, sample_ids: list[str], anchor_count: int) -> set[str]:
        if anchor_count <= 0:
            return set()
        ordered = sorted(set(sample_ids))
        return set(ordered[: min(len(ordered), int(anchor_count))])

    def _write_expert_manifests(
        self,
        *,
        pack_dir: Path,
        pack_id_prefix: str,
        public_items: list[dict],
        private_items: list[dict],
        expert_count: int,
    ) -> list[str]:
        manifest_dir = pack_dir / "expert_manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        created_files: list[str] = []
        private_lookup = {item["sample_id"]: item for item in private_items}

        for expert_idx in range(1, max(1, expert_count) + 1):
            rng = random.Random(self.seed + expert_idx)
            ordered_public = list(public_items)
            rng.shuffle(ordered_public)
            ordered_private = [private_lookup[x["sample_id"]] for x in ordered_public if x["sample_id"] in private_lookup]
            expert_pack_id = f"{pack_id_prefix}_E{expert_idx:02d}"
            manifest_public = {
                "pack_id": expert_pack_id,
                "expert_index": expert_idx,
                "items": ordered_public,
            }
            manifest_private = {
                "pack_id": expert_pack_id,
                "expert_index": expert_idx,
                "items": ordered_private,
            }
            pub_path = manifest_dir / f"manifest_public_E{expert_idx:02d}.json"
            pri_path = manifest_dir / f"manifest_private_E{expert_idx:02d}.json"
            pub_path.write_text(json.dumps(manifest_public, indent=2), encoding="utf-8")
            pri_path.write_text(json.dumps(manifest_private, indent=2), encoding="utf-8")
            created_files.extend([str(pub_path), str(pri_path)])

        return created_files

    # ------------------------------------------------------------------
    # Top-1 vs Real  (V7 default)
    # ------------------------------------------------------------------
    def _run_top1_vs_real(self, selected_ids: list[str], meta_lookup: dict[str, dict]):
        """Build a blinded A/B pack: restored (top-1 method) vs real image."""
        top1_method = self._load_top1_method()
        self.logger.info(f"Top-1 method from composite ranking: {top1_method}")

        pack_dir = self.output_dir / "Expert_Pack_Top1vsReal"
        if pack_dir.exists():
            shutil.rmtree(pack_dir, ignore_errors=True)
        pack_dir.mkdir(parents=True, exist_ok=True)
        images_root = pack_dir / "images"
        images_root.mkdir(parents=True, exist_ok=True)

        cond_slug = self._slug(self.condition)

        public_items: list[dict] = []
        private_items: list[dict] = []
        skipped = {"missing_sample_dir": 0, "missing_input": 0, "missing_restored": 0, "missing_gt": 0}

        for sid in selected_ids:
            src_dir = self._resolve_sample_dir(sid)
            if src_dir is None:
                skipped["missing_sample_dir"] += 1
                continue

            input_path = src_dir / "masked_input.png"
            if not input_path.exists():
                skipped["missing_input"] += 1
                continue

            # Restored output from the top-1 method
            restored_path = src_dir / f"{self._slug(top1_method)}_{cond_slug}.png"
            if not restored_path.exists():
                skipped["missing_restored"] += 1
                continue

            # Real undamaged GT
            gt_path = self._find_ground_truth(sid)
            if gt_path is None:
                skipped["missing_gt"] += 1
                continue

            dest_dir = images_root / sid
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, dest_dir / "input.png")

            # Blinded assignment: randomly label restored/real as A or B
            sid_hash = int(md5(sid.encode("utf-8", errors="ignore")).hexdigest()[:8], 16)
            local_rng = random.Random(self.seed ^ sid_hash)
            order = [
                ("restored", top1_method, restored_path),
                ("real", "Ground-Truth", gt_path),
            ]
            local_rng.shuffle(order)

            mapping: dict[str, str] = {}
            for label, (_, source_name, path) in zip(["A", "B"], order):
                shutil.copy2(path, dest_dir / f"{label}.png")
                mapping[label] = source_name

            public_items.append(
                {
                    "sample_id": sid,
                    "input": f"images/{sid}/input.png",
                    "A": f"images/{sid}/A.png",
                    "B": f"images/{sid}/B.png",
                    "is_anchor": False,
                    **(meta_lookup.get(sid, {})),
                }
            )
            private_items.append({"sample_id": sid, "mapping": mapping})

        ev_cfg = self.config.get("expert_validation", {}) if self.config else {}
        anchor_count = int(getattr(ev_cfg, "anchor_samples", 4) or 4)
        expert_count = int(getattr(ev_cfg, "expert_count", 1) or 1)
        anchor_ids = self._anchor_sample_ids([x["sample_id"] for x in public_items], anchor_count)
        for item in public_items:
            item["is_anchor"] = bool(item["sample_id"] in anchor_ids)
        for item in private_items:
            item["is_anchor"] = bool(item["sample_id"] in anchor_ids)

        self._write_manifests(
            pack_dir=pack_dir,
            pack_id="Expert_Pack_Top1vsReal",
            public_items=public_items,
            private_items=private_items,
            extra_meta={
                "mode": "top1_vs_real",
                "top1_method": top1_method,
                "condition": self.condition,
            },
            rating_task=(
                "Given the damaged input shown above, which candidate is the more plausible "
                "restoration of the missing region?"
            ),
        )

        summary = {
            "mode": "top1_vs_real",
            "top1_method": top1_method,
            "selected_ids": len(selected_ids),
            "created_items": len(public_items),
            "anchor_items": len(anchor_ids),
            "expert_count": expert_count,
            "practice_samples": self.practice_samples,
            "skipped": skipped,
            "condition": self.condition,
            "pack_dir": str(pack_dir),
        }
        created_expert_manifests = self._write_expert_manifests(
            pack_dir=pack_dir,
            pack_id_prefix="Expert_Pack_Top1vsReal",
            public_items=public_items,
            private_items=private_items,
            expert_count=expert_count,
        )
        summary["expert_manifest_files"] = created_expert_manifests
        (self.output_dir / "stage_18_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        if not public_items:
            raise RuntimeError(
                f"Stage 18 (top1_vs_real) produced zero items. Summary: {summary}"
            )
        self.logger.info(
            f"Stage 18 (top1_vs_real) complete. {len(public_items)} blinded items in {pack_dir}."
        )

    # ------------------------------------------------------------------
    # Legacy method-pair mode
    # ------------------------------------------------------------------
    def _run_method_pair(self, selected_ids: list[str], meta_lookup: dict[str, dict]):
        """Build a blinded A/B pack comparing two named restoration methods."""
        pack_dir = self.output_dir / "Expert_Pack_v2"
        if pack_dir.exists():
            shutil.rmtree(pack_dir, ignore_errors=True)
        pack_dir.mkdir(parents=True, exist_ok=True)
        images_root = pack_dir / "images"
        images_root.mkdir(parents=True, exist_ok=True)

        method_a, method_b = (self.method_pair + ["Telea", "FT-SD"])[:2]
        cond_slug = self._slug(self.condition)

        public_items: list[dict] = []
        private_items: list[dict] = []
        skipped = {"missing_sample_dir": 0, "missing_input": 0, "missing_outputs": 0}

        for sid in selected_ids:
            src_dir = self._resolve_sample_dir(sid)
            if src_dir is None:
                skipped["missing_sample_dir"] += 1
                continue

            input_path = src_dir / "masked_input.png"
            if not input_path.exists():
                skipped["missing_input"] += 1
                continue

            a_path = src_dir / f"{self._slug(method_a)}_{cond_slug}.png"
            b_path = src_dir / f"{self._slug(method_b)}_{cond_slug}.png"
            if not a_path.exists() or not b_path.exists():
                skipped["missing_outputs"] += 1
                continue

            dest_dir = images_root / sid
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, dest_dir / "input.png")

            sid_hash = int(md5(sid.encode("utf-8", errors="ignore")).hexdigest()[:8], 16)
            local_rng = random.Random(self.seed ^ sid_hash)
            order = [("A", method_a, a_path), ("B", method_b, b_path)]
            local_rng.shuffle(order)

            mapping: dict[str, str] = {}
            for label, method_name, path in order:
                shutil.copy2(path, dest_dir / f"{label}.png")
                mapping[label] = method_name

            public_items.append(
                {
                    "sample_id": sid,
                    "input": f"images/{sid}/input.png",
                    "A": f"images/{sid}/A.png",
                    "B": f"images/{sid}/B.png",
                    **(meta_lookup.get(sid, {})),
                }
            )
            private_items.append({"sample_id": sid, "mapping": mapping})

        self._write_manifests(
            pack_dir=pack_dir,
            pack_id="Expert_Pack_v2",
            public_items=public_items,
            private_items=private_items,
            extra_meta={
                "mode": "method_pair",
                "method_pair": [method_a, method_b],
                "condition": self.condition,
            },
            rating_task=(
                "Pick the better restoration for the missing region "
                "(historical plausibility + visual coherence)."
            ),
        )

        # Backwards-compatible mapping file (PRIVATE)
        mapping_all = {it["sample_id"]: it["mapping"] for it in private_items}
        (self.output_dir / "method_mapping.json").write_text(
            json.dumps(mapping_all, indent=2), encoding="utf-8"
        )

        summary = {
            "mode": "method_pair",
            "selected_ids": len(selected_ids),
            "created_items": len(public_items),
            "skipped": skipped,
            "method_pair": [method_a, method_b],
            "condition": self.condition,
            "pack_dir": str(pack_dir),
        }
        (self.output_dir / "stage_18_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        if not public_items:
            raise RuntimeError(
                f"Stage 18 (method_pair) produced zero items. Summary: {summary}"
            )
        self.logger.info(
            f"Stage 18 (method_pair) complete. {len(public_items)} items in {pack_dir}."
        )

    # ------------------------------------------------------------------
    # Manifest writer (shared)
    # ------------------------------------------------------------------
    def _write_manifests(
        self,
        *,
        pack_dir: Path,
        pack_id: str,
        public_items: list[dict],
        private_items: list[dict],
        extra_meta: dict,
        rating_task: str,
    ):
        created_at = datetime.now(timezone.utc).isoformat()
        rating_schema = {
            "task": rating_task,
            "fields": [
                {"name": "choice", "type": "enum", "values": ["A", "B", "Tie", "Unsure"]},
                {"name": "confidence", "type": "int", "min": 1, "max": 5},
                {"name": "comment", "type": "string", "optional": True},
            ],
        }
        manifest_public = {
            "pack_id": pack_id,
            "created_at": created_at,
            **extra_meta,
            "rating_schema": rating_schema,
            "items": public_items,
        }
        manifest_private = {
            "pack_id": pack_id,
            "created_at": created_at,
            "items": private_items,
        }
        (pack_dir / "manifest_public.json").write_text(
            json.dumps(manifest_public, indent=2), encoding="utf-8"
        )
        (pack_dir / "manifest_private.json").write_text(
            json.dumps(manifest_private, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(self):
        self.logger.info("=" * 20 + " STAGE 18: Expert Validation Pack " + "=" * 20)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.samples_dir.exists():
            self.logger.warning(
                f"Evaluation samples not found at {self.samples_dir}. Skipping S18."
            )
            return

        selected_ids = self._select_samples()
        if not selected_ids:
            self.logger.warning("No samples selected for expert validation. Skipping S18.")
            return

        meta_lookup = self._build_meta_lookup()

        if self.mode == "top1_vs_real":
            self._run_top1_vs_real(selected_ids, meta_lookup)
        else:
            self._run_method_pair(selected_ids, meta_lookup)


if __name__ == "__main__":
    cm = ConfigManager()
    stage = ExpertValidationStage(cm)
    stage.run()
