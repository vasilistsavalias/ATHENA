import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from thesis_pipeline.components.data_acquisition import DataAcquisition
from thesis_pipeline.components.data_acquisition_europeana import EuropeanaOpenAcquirer
from thesis_pipeline.components.data_cleaning import DataCleaner
from thesis_pipeline.utils.env_loader import load_env_file


class DataAcquisitionStage:
    """Stage S02 - Data acquisition with integrated cleaning."""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.config = config_manager.get_data_acquisition_config()
        self.paths = config_manager.get_paths()
        self.logger = logging.getLogger(__name__)

        self.raw_dir = Path(self.paths.data.raw)
        self.wikimedia_dir = self.raw_dir / "wikimedia"
        self.europeana_dir = self.raw_dir / "europeana"
        self.artifacts_dir = self.config_manager.get_stage_artifact_dir("S02")
        self.filtered_dir = Path(self.paths.data.filtered)

    @staticmethod
    def _image_manifest(root: Path) -> list[dict[str, str]]:
        if not root.exists():
            return []
        patterns = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.webp")
        rows: list[dict[str, str]] = []
        for pattern in patterns:
            for path in sorted(root.rglob(pattern)):
                rows.append(
                    {
                        "id": path.stem,
                        "filename": path.name,
                        "relative_path": str(path.relative_to(root)),
                    }
                )
        return rows

    def _run_integrated_cleaning(self) -> None:
        """Run the former S02b cleaning logic as an internal S02 step."""
        try:
            report_dir = self.artifacts_dir / "cleaning"
            report_dir.mkdir(parents=True, exist_ok=True)
            self.filtered_dir.mkdir(parents=True, exist_ok=True)

            dc = getattr(self.config_manager.config, "data_cleaning", {})
            min_width = int(getattr(dc, "min_width", 0) or 0)
            min_height = int(getattr(dc, "min_height", 0) or 0)
            color_check = bool(getattr(dc, "color_check", True))
            saturation_threshold = float(getattr(dc, "saturation_threshold", 20.0) or 20.0)
            whiteness_threshold = float(getattr(dc, "whiteness_threshold", 0.85) or 0.85)

            cleaner = DataCleaner(
                input_dir=self.raw_dir,
                filtered_dir=self.filtered_dir,
                extensions=[".jpg", ".jpeg", ".png"],
            )
            cleaner.filter_grayscale_images(
                report_dir=report_dir,
                sample_limit=50,
                min_width=min_width,
                min_height=min_height,
                color_check=color_check,
                saturation_threshold=saturation_threshold,
                whiteness_threshold=whiteness_threshold,
                seed=42,
            )
            self.logger.info("Integrated S02 cleaning completed.")
        except Exception as exc:
            self.logger.exception(f"Integrated cleaning failed: {exc}")
            raise

    @staticmethod
    def _count_by_prefix(raw_dir: Path, prefix: str) -> int:
        if not raw_dir.exists():
            return 0
        patterns = [
            f"{prefix}_*.jpg",
            f"{prefix}_*.jpeg",
            f"{prefix}_*.png",
            f"{prefix}_*.tif",
            f"{prefix}_*.tiff",
            f"{prefix}_*.webp",
        ]
        count = 0
        for pattern in patterns:
            count += len(list(raw_dir.glob(pattern)))
        return count

    def _iter_all_metadata_json(self):
        for meta_path in self.raw_dir.rglob("metadata/*.json"):
            yield meta_path

    @staticmethod
    def _evaluate_validity_violations(
        *,
        counts_after: dict[str, int],
        total_after: int,
        wiki_enabled: bool,
        eur_enabled: bool,
        min_total_images: int,
        require_enabled_source_nonzero: bool,
        require_europeana_key_when_enabled: bool,
        europeana_key_present: bool,
    ) -> list[str]:
        violations: list[str] = []
        if min_total_images > 0 and total_after < min_total_images:
            violations.append(
                f"total images {total_after} below configured minimum {min_total_images}"
            )

        if require_enabled_source_nonzero:
            if wiki_enabled and counts_after.get("wikimedia", 0) <= 0:
                violations.append("wikimedia_enabled=true but downloaded count is zero")
            if eur_enabled and counts_after.get("europeana", 0) <= 0:
                violations.append("europeana_enabled=true but downloaded count is zero")

        if require_europeana_key_when_enabled and eur_enabled and not europeana_key_present:
            violations.append("europeana_enabled=true but EUROPEANA_API_KEY is missing")

        return violations

    def _migrate_legacy_flat_layout(self) -> None:
        prefix_to_dir = {
            "wiki_": self.wikimedia_dir,
            "eur_": self.europeana_dir,
        }

        try:
            for path in self.raw_dir.iterdir():
                if not path.is_file():
                    continue
                target_dir = None
                for pref, dest_dir in prefix_to_dir.items():
                    if path.name.startswith(pref):
                        target_dir = dest_dir
                        break
                if not target_dir:
                    continue
                target_dir.mkdir(parents=True, exist_ok=True)
                dest = target_dir / path.name
                if dest.exists():
                    stem = dest.stem
                    suffix = dest.suffix
                    index = 1
                    while True:
                        alt = target_dir / f"{stem}_dup{index}{suffix}"
                        if not alt.exists():
                            dest = alt
                            break
                        index += 1
                shutil.move(str(path), str(dest))
        except Exception:
            pass

        legacy_meta = self.raw_dir / "metadata"
        if legacy_meta.exists() and legacy_meta.is_dir():
            try:
                for meta in legacy_meta.glob("*.json"):
                    target_dir = None
                    for pref, dest_dir in prefix_to_dir.items():
                        if meta.name.startswith(pref):
                            target_dir = dest_dir / "metadata"
                            break
                    if not target_dir:
                        continue
                    target_dir.mkdir(parents=True, exist_ok=True)
                    dest = target_dir / meta.name
                    if dest.exists():
                        try:
                            meta.unlink()
                        except Exception:
                            pass
                        continue
                    shutil.move(str(meta), str(dest))
                try:
                    if not any(legacy_meta.iterdir()):
                        legacy_meta.rmdir()
                except Exception:
                    pass
            except Exception:
                pass

    def run(self):
        self.logger.info("=" * 20 + " STAGE S02: Data Acquisition + Cleaning " + "=" * 20)

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.wikimedia_dir.mkdir(parents=True, exist_ok=True)
        self.europeana_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy_flat_layout()

        try:
            cfg_path = Path(getattr(self.config_manager, "config_filepath", "") or "").resolve()
            if cfg_path.exists():
                project_root = cfg_path.parents[2]
            else:
                project_root = Path(__file__).resolve().parents[3]
            loaded_keys = load_env_file(project_root / ".env")
            if loaded_keys:
                self.logger.info(f"Loaded env keys from .env: {', '.join(sorted(set(loaded_keys)))}")
        except Exception:
            pass

        max_per_source = int(getattr(self.config, "max_per_source", 10_000) or 10_000)
        validity_cfg = getattr(self.config, "validity", {})
        min_total_images = int(getattr(validity_cfg, "min_total_images", 0) or 0)
        require_enabled_source_nonzero = bool(
            getattr(validity_cfg, "require_enabled_source_nonzero", True)
        )
        require_europeana_key_when_enabled = bool(
            getattr(validity_cfg, "require_europeana_key_when_enabled", True)
        )
        pipeline_cfg = self.config_manager.config.get("pipeline", {})
        strict_fail_policy = bool(pipeline_cfg.get("strict_fail_policy", False))

        wiki_enabled = bool(getattr(self.config, "wikimedia_enabled", True))
        wiki_limit = int(getattr(self.config, "wikimedia_limit", max_per_source) or max_per_source)
        wiki_query = str(getattr(self.config, "start_category", "Ancient Greek pottery") or "Ancient Greek pottery")
        wiki_api = str(getattr(self.config, "wikimedia_api_url", "https://commons.wikimedia.org/w/api.php"))

        eur_enabled = bool(getattr(self.config, "europeana_enabled", True))
        eur_limit = int(getattr(self.config, "europeana_limit", max_per_source) or max_per_source)
        eur_query = str(getattr(self.config, "europeana_query", "ancient greek vase") or "ancient greek vase")
        eur_query_variants = getattr(self.config, "europeana_query_variants", None)
        eur_rows = int(getattr(self.config, "europeana_rows", 100) or 100)
        eur_reuse = str(getattr(self.config, "europeana_reusability", "open") or "open")
        eur_qf_filters = getattr(self.config, "europeana_qf_filters", None)
        eur_collection_ids = getattr(self.config, "europeana_collection_ids", None)
        eur_copyright_free_only = bool(getattr(self.config, "europeana_copyright_free_only", True))
        eur_rights_allowlist = getattr(self.config, "europeana_additional_copyright_free_rights", None)
        eur_reset_state_on_run = bool(getattr(self.config, "europeana_reset_state_on_run", False))
        eur_backup_state_before_reset = bool(getattr(self.config, "europeana_backup_state_before_reset", True))
        extra_qf_filters: list[str] = []
        if isinstance(eur_qf_filters, list):
            extra_qf_filters.extend([str(f).strip() for f in eur_qf_filters if str(f).strip()])
        if isinstance(eur_collection_ids, list):
            extra_qf_filters.extend([f"DATASET:{str(c).strip()}" for c in eur_collection_ids if str(c).strip()])
        eur_queries: list[str] = []
        if isinstance(eur_query_variants, list):
            eur_queries = [str(q).strip() for q in eur_query_variants if str(q).strip()]
        if not eur_queries:
            eur_queries = [eur_query]

        summaries: list[dict] = []
        failures: list[dict] = []

        counts_before = {
            "wikimedia": self._count_by_prefix(self.wikimedia_dir, "wiki"),
            "europeana": self._count_by_prefix(self.europeana_dir, "eur"),
        }

        eur_state_file = self.europeana_dir / "europeana_state.json"
        if eur_enabled and eur_reset_state_on_run and eur_state_file.exists():
            try:
                if eur_backup_state_before_reset:
                    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    backup = self.europeana_dir / f"europeana_state.backup_{ts}.json"
                    shutil.copy2(eur_state_file, backup)
                    self.logger.info(f"Europeana state backup created: {backup}")
                eur_state_file.unlink(missing_ok=True)
                self.logger.info("Europeana state reset for this run (europeana_reset_state_on_run=true).")
            except Exception as exc:
                self.logger.warning(f"Failed to reset Europeana state file: {exc}")

        def run_wikimedia():
            if not wiki_enabled:
                return {"source": "wikimedia", "status": "skipped"}
            if counts_before["wikimedia"] >= wiki_limit:
                return {"source": "wikimedia", "status": "skipped", "reason": "limit_met"}
            self.logger.info(
                f"Wikimedia: downloading up to {wiki_limit} images "
                f"(current={counts_before['wikimedia']}) -> {self.wikimedia_dir}"
            )
            DataAcquisition(api_url=wiki_api).download_images_from_category(
                start_category=wiki_query,
                output_dir=self.wikimedia_dir,
                limit=wiki_limit,
                filename_prefix="wiki",
                state_file_name="scraper_state_wiki.json",
            )
            return {"source": "wikimedia", "status": "ok"}

        def run_europeana():
            if not eur_enabled:
                return {"source": "europeana", "status": "skipped"}
            if counts_before["europeana"] >= eur_limit:
                return {"source": "europeana", "status": "skipped", "reason": "limit_met"}
            import os

            api_key = os.environ.get("EUROPEANA_API_KEY", "").strip()
            self.logger.info(
                f"Europeana: downloading up to {eur_limit} images "
                f"(current={counts_before['europeana']}) [key={'set' if api_key else 'missing'}] -> {self.europeana_dir}"
            )
            acquirer = EuropeanaOpenAcquirer(api_key=api_key)
            per_query: list[dict] = []
            current_count = counts_before["europeana"]

            for query in eur_queries:
                remaining = max(0, eur_limit - current_count)
                if remaining <= 0:
                    break

                before_query = current_count
                eur_summary = acquirer.download(
                    query=query,
                    output_dir=self.europeana_dir,
                    metadata_dir=self.europeana_dir / "metadata",
                    limit=remaining,
                    rows=eur_rows,
                    reusability=eur_reuse,
                    qf_filters=extra_qf_filters,
                    copyright_free_only=eur_copyright_free_only,
                    additional_copyright_free_rights=eur_rights_allowlist,
                    state_file=eur_state_file,
                )
                current_count = self._count_by_prefix(self.europeana_dir, "eur")
                per_query.append(
                    {
                        "query": query,
                        "attempted": int(eur_summary.attempted),
                        "downloaded": int(eur_summary.downloaded),
                        "downloaded_delta": int(max(0, current_count - before_query)),
                        "skipped_no_full_image": int(eur_summary.skipped_no_full_image),
                        "skipped_no_rights": int(eur_summary.skipped_no_rights),
                        "skipped_non_copyright_free": int(eur_summary.skipped_non_copyright_free),
                        "failed_downloads": int(eur_summary.failed_downloads),
                        "errors": list(eur_summary.errors),
                    }
                )
                if current_count >= eur_limit:
                    break

            return {
                "source": "europeana",
                "status": "ok",
                "query_runs": per_query,
                "final_count": current_count,
            }

        futures = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures[executor.submit(run_wikimedia)] = "wikimedia"
            futures[executor.submit(run_europeana)] = "europeana"

            for future in as_completed(futures):
                source = futures[future]
                try:
                    summaries.append(future.result())
                except Exception as exc:
                    self.logger.warning(f"{source} acquisition failed: {exc}")
                    failures.append({"source": source, "error": str(exc)})

        counts_after = {
            "wikimedia": self._count_by_prefix(self.wikimedia_dir, "wiki"),
            "europeana": self._count_by_prefix(self.europeana_dir, "eur"),
        }
        total_after = sum(counts_after.values())

        validity_violations = self._evaluate_validity_violations(
            counts_after=counts_after,
            total_after=total_after,
            wiki_enabled=wiki_enabled,
            eur_enabled=eur_enabled,
            min_total_images=min_total_images,
            require_enabled_source_nonzero=require_enabled_source_nonzero,
            require_europeana_key_when_enabled=require_europeana_key_when_enabled,
            europeana_key_present=bool((os.environ.get("EUROPEANA_API_KEY") or "").strip()),
        )

        if total_after <= 0:
            raise RuntimeError(f"Stage 02 failed: zero images across all sources. failures={failures}")

        self._run_integrated_cleaning()

        try:
            acquired_manifest = self._image_manifest(self.raw_dir)
            cleaned_manifest = self._image_manifest(self.filtered_dir)
            source_counts = [
                {"source": "wikimedia", "count": counts_after["wikimedia"]},
                {"source": "europeana", "count": counts_after["europeana"]},
                {"source": "total", "count": total_after},
            ]
            (self.artifacts_dir / "source_counts.csv").write_text(
                "source,count\n" + "\n".join(f"{r['source']},{r['count']}" for r in source_counts) + "\n",
                encoding="utf-8",
            )

            try:
                import matplotlib.pyplot as plt

                labels = ["Wikimedia", "Europeana"]
                values = [counts_after["wikimedia"], counts_after["europeana"]]
                plt.figure(figsize=(8, 4), dpi=180)
                bars = plt.bar(labels, values, color=["#1f77b4", "#ff7f0e"], edgecolor="black")
                for bar, value in zip(bars, values):
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        value + max(values) * 0.01 if max(values) > 0 else value,
                        f"{value:,}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )
                plt.title("Downloaded Images by Source (Raw)")
                plt.ylabel("count")
                plt.tight_layout()
                plt.savefig(self.artifacts_dir / "source_counts.png", bbox_inches="tight")
                plt.close()
            except Exception as exc:
                self.logger.info(f"Stage 02 bar chart skipped: {exc}")

            try:
                query_rows: list[dict] = []
                for summary in summaries:
                    if summary.get("source") != "europeana":
                        continue
                    for row in summary.get("query_runs", []) or []:
                        query_rows.append(
                            {
                                "source": "europeana",
                                "query": str(row.get("query", "")),
                                "attempted": int(row.get("attempted", 0) or 0),
                                "downloaded": int(row.get("downloaded", 0) or 0),
                                "downloaded_delta": int(row.get("downloaded_delta", 0) or 0),
                                "skipped_no_full_image": int(row.get("skipped_no_full_image", 0) or 0),
                                "skipped_no_rights": int(row.get("skipped_no_rights", 0) or 0),
                                "failed_downloads": int(row.get("failed_downloads", 0) or 0),
                            }
                        )
                if query_rows:
                    header = list(query_rows[0].keys())
                    lines = [",".join(header)]
                    for row in query_rows:
                        lines.append(
                            ",".join(
                                [
                                    str(row["source"]).replace(",", ";"),
                                    str(row["query"]).replace(",", ";"),
                                    str(row["attempted"]),
                                    str(row["downloaded"]),
                                    str(row["downloaded_delta"]),
                                    str(row["skipped_no_full_image"]),
                                    str(row["skipped_no_rights"]),
                                    str(row["failed_downloads"]),
                                ]
                            )
                        )
                    (self.artifacts_dir / "europeana_query_yield.csv").write_text(
                        "\n".join(lines) + "\n",
                        encoding="utf-8",
                    )
            except Exception as exc:
                self.logger.info(f"Europeana query-yield artifact skipped: {exc}")

            try:
                licenses: dict[str, int] = {}
                for meta_path in self._iter_all_metadata_json():
                    try:
                        data = json.loads(meta_path.read_text(encoding="utf-8", errors="ignore"))
                        license_name = str(data.get("license") or "").strip()
                        if not license_name:
                            continue
                        licenses[license_name] = licenses.get(license_name, 0) + 1
                    except Exception:
                        continue
                if licenses:
                    license_rows = sorted(licenses.items(), key=lambda kv: kv[1], reverse=True)
                    (self.artifacts_dir / "license_summary.csv").write_text(
                        "license,count\n" + "\n".join(f"{license_name.replace(',', ';')},{count}" for license_name, count in license_rows) + "\n",
                        encoding="utf-8",
                    )
            except Exception as exc:
                self.logger.info(f"License summary skipped: {exc}")

            summary_doc = {
                "stage": "02_data_acquisition",
                "raw_dir": str(self.raw_dir),
                "source_dirs": {
                    "wikimedia": str(self.wikimedia_dir),
                    "europeana": str(self.europeana_dir),
                },
                "query_variants": {
                    "europeana": eur_queries,
                },
                "counts_before": counts_before,
                "counts_after": counts_after,
                "summaries": summaries,
                "failures": failures,
                "validity": {
                    "strict_fail_policy": strict_fail_policy,
                    "min_total_images": min_total_images,
                    "require_enabled_source_nonzero": require_enabled_source_nonzero,
                    "require_europeana_key_when_enabled": require_europeana_key_when_enabled,
                    "violations": validity_violations,
                },
            }
            (self.artifacts_dir / "download_summary.json").write_text(
                json.dumps(summary_doc, indent=2),
                encoding="utf-8",
            )
            (self.artifacts_dir / "acquisition_validity.json").write_text(
                json.dumps(summary_doc["validity"], indent=2),
                encoding="utf-8",
            )
            (self.artifacts_dir / "acquired_image_manifest.json").write_text(
                json.dumps(
                    {
                        "count": len(acquired_manifest),
                        "images": acquired_manifest,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (self.artifacts_dir / "cleaned_image_manifest.json").write_text(
                json.dumps(
                    {
                        "count": len(cleaned_manifest),
                        "images": cleaned_manifest,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            self.logger.warning(f"Failed to write Stage 02 artifacts: {exc}")

        if validity_violations:
            message = " | ".join(validity_violations)
            if strict_fail_policy:
                raise RuntimeError(f"Stage 02 validity gate failed (strict): {message}")
            self.logger.warning(f"Stage 02 validity gate warnings: {message}")

        self.logger.info(f"Stage 02 complete. Counts: {counts_after} (total={total_after})")


if __name__ == "__main__":
    from thesis_pipeline.config_manager import ConfigManager

    config_manager = ConfigManager()
    stage = DataAcquisitionStage(config_manager)
    stage.run()
