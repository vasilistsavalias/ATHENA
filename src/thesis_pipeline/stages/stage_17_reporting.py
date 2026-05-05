# src/thesis_pipeline/stages/stage_17_reporting.py
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import json
from PIL import Image
from thesis_pipeline.config_manager import ConfigManager
from thesis_pipeline.visualization import ThesisPlotter
from thesis_pipeline.reporting.odyssey_generator import OdysseyReportGenerator

class ReportingStage:
    SAMPLE_MODEL_FILE_MAP = {
        "Telea": "Telea_Unconditional.png",
        "Navier-Stokes": "Navier_Stokes_Unconditional.png",
        "Vanilla SD": "Vanilla_SD_Unconditional.png",
        "FT-SD": "FT_SD_Unconditional.png",
        "FT-SD+TTA": "FT_SD_TTA_Unconditional.png",
        "LaMa": "LaMa_Unconditional.png",
        "MAT": "MAT_Unconditional.png",
        "CoModGAN": "CoModGAN_Unconditional.png",
    }

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.config = config_manager.config
        self.paths = config_manager.get_paths()
        self.logger = logging.getLogger(__name__)
        self.output_dir = self.config_manager.get_stage_artifact_dir("S17")

    def run(self):
        self.logger.info("="*20 + " STAGE 17: Final Reporting " + "="*20)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            plotter = ThesisPlotter(self.output_dir)
            
            # 1. Execution Times
            log_path = Path(self.paths.artifacts.logs) / "execution_times.csv"
            if log_path.exists():
                plotter.plot_execution_time(log_path, "pipeline_execution_times")

            # 2. Evaluation metric charts (from Stage 15 CSVs)
            self._generate_evaluation_charts(plotter)

            # 3. Odyssey
            hero_config = self.config.get("hero_tracking")
            if hero_config and hero_config.get("enabled"):
                hero_dir = Path(hero_config.output_dir)
                if hero_dir.exists():
                    OdysseyReportGenerator(hero_dir).generate_report()

            self.logger.info("="*20 + " STAGE 17 COMPLETED " + "="*20 + "\n")
        except Exception as e:
            self.logger.exception(f"Error: {e}")

    def _generate_evaluation_charts(self, plotter: ThesisPlotter):
        """Re-generate evaluation charts from Stage 15 CSV outputs.

        This allows the reporting stage to produce publication-ready charts
        even when Stage 15 ran on a different machine (e.g. GCP)."""
        stage13 = self.config_manager.get_stage_artifact_dir("S15")
        stage06 = self.config_manager.get_stage_artifact_dir("S07")
        stage12 = self.config_manager.get_stage_artifact_dir("S13")
        matrix_csv = stage13 / "benchmarking_matrix" / "matrix_results.csv"
        stats_csv = stage13 / "statistical_tests" / "paired_t_tests.csv"

        if not matrix_csv.exists():
            self.logger.info("No Stage 15 matrix_results.csv found — skipping evaluation charts.")
            return

        self.logger.info("Generating evaluation charts from Stage 15 data…")
        try:
            df = pd.read_csv(matrix_csv)
            if "severity_bin" not in df.columns and "mask_coverage" in df.columns:
                df["severity_bin"] = pd.cut(
                    df["mask_coverage"],
                    bins=[0, 0.1, 0.25, 0.5, 1.0],
                    labels=["<10%", "10-25%", "25-50%", ">50%"],
                    include_lowest=True,
                )

            # --- Combined overview charts ---
            plotter.plot_metric_bars(df, filename="metric_comparison_bars")
            plotter.plot_metric_distributions(df, filename="metric_distributions")
            if 'mask_coverage' in df.columns:
                plotter.plot_psnr_vs_coverage(df, filename="psnr_vs_coverage")
            plotter.plot_improvement_deltas(df, metric='psnr', baseline='Telea',
                                           filename="improvement_over_telea")
            if stats_csv.exists():
                stats_df = pd.read_csv(stats_csv)
                plotter.plot_significance_heatmap(stats_df, filename="significance_heatmap")
                sig_matrix_cfg = self.config.model_evaluation.get("significance_matrix", {})
                matrix_plotter = ThesisPlotter(self.output_dir / "charts")
                manifest_df = matrix_plotter.plot_significance_matrix_suite(
                    stats_df=stats_df,
                    config=sig_matrix_cfg,
                )
                if not manifest_df.empty:
                    generated = int((manifest_df["status"] == "generated").sum())
                    skipped = int((manifest_df["status"] == "skipped").sum())
                    self.logger.info(
                        "Significance matrix charts regenerated in Stage 17: "
                        f"{generated} generated, {skipped} skipped "
                        f"(manifest: {matrix_plotter.output_dir / 'significance_matrix' / 'matrix_manifest.csv'})."
                    )
            else:
                stats_df = None

            # --- Individual per-metric charts ---
            self.logger.info("Generating individual per-metric charts…")
            plotter.plot_all_individual_charts(df, stats_df)

            # --- Ablation study charts ---
            self.logger.info("Generating ablation study charts…")
            for model_name in ['FT-SD', 'Vanilla SD']:
                if model_name in df['model'].unique():
                    plotter.plot_ablation_bars(df, model=model_name)
            plotter.plot_ablation_heatmap(df)
            for metric in ['psnr', 'ssim', 'lpips', 'color', 'pattern']:
                if metric in df.columns:
                    plotter.plot_ablation_per_metric(df, metric)

            # --- Training analysis charts ---
            training_csv = self.config_manager.get_stage_artifact_path("S13", "training_logs.csv")
            if training_csv.exists():
                self.logger.info("Generating training analysis charts…")
                plotter.plot_training_analysis(str(training_csv))
                plotter.plot_loss_landscape(str(training_csv))

            # --- Sweep analysis charts ---
            stage12 = self.config_manager.get_stage_artifact_dir("S13")
            sweep_summary_csv = stage12 / "sweep" / "trials_summary.csv"
            if sweep_summary_csv.exists():
                sweep_df = pd.read_csv(sweep_summary_csv)
                plotter.plot_sweep_pareto(sweep_df)
                plotter.plot_lr_wd_response_heatmap(sweep_df)
                plotter.plot_trial_learning_curves_panel(stage12 / "sweep")

            # --- K-fold training stability chart ---
            kfold_train_csv = stage12 / "kfold_training_summary.csv"
            if kfold_train_csv.exists():
                kfold_df = pd.read_csv(kfold_train_csv)
                plotter.plot_kfold_training_stability(kfold_df)

            # --- Hyperparameter summary ---
            hp_yaml = self.config_manager.get_stage_artifact_path("S12", "best_hyperparameters.yaml")
            if hp_yaml.exists():
                import yaml
                with open(hp_yaml) as f:
                    hp = yaml.safe_load(f)
                plotter.plot_hyperparameter_summary(hp)

            # --- LR schedule ---
            plotter.plot_lr_schedule()

            # --- Mask-type ablation ---
            if 'mask_type' in df.columns and df['mask_type'].nunique() > 1:
                plotter.plot_mask_type_ablation(df)
                plotter.plot_mask_type_heatmap(df)
            if 'mask_coverage' in df.columns:
                coverage_bins = pd.cut(
                    df['mask_coverage'],
                    bins=[0, 0.1, 0.25, 0.5, 1.0],
                    labels=['<10%', '10-25%', '25-50%', '>50%'],
                    include_lowest=True,
                )
                if coverage_bins.nunique() > 1:
                    df = df.copy()
                    df['coverage_bin'] = coverage_bins
                    plotter.plot_stratified_bars(
                        df,
                        group_col='coverage_bin',
                        filename='stratified_by_coverage',
                        title='PSNR by Mask Coverage Bin',
                    )
            if 'severity_bin' in df.columns and df['severity_bin'].nunique() > 1:
                plotter.plot_stratified_bars(
                    df,
                    group_col='severity_bin',
                    filename='stratified_by_severity',
                    title='PSNR by Damage Severity Bin',
                )

            # --- TTA comparison ---
            if 'FT-SD+TTA' in df['model'].unique():
                plotter.plot_tta_comparison(df)

            # --- Inference timing ---
            timing_csv = stage13 / "benchmarking_matrix" / "inference_timing.csv"
            if timing_csv.exists():
                timing_df = pd.read_csv(timing_csv)
                plotter.plot_inference_timing(timing_df)

            # --- FID/KID ---
            fid_kid_csv = stage13 / "benchmarking_matrix" / "fid_kid_scores.csv"
            if fid_kid_csv.exists():
                fid_kid_df = pd.read_csv(fid_kid_csv)
                fid_scores = dict(zip(fid_kid_df['model'], fid_kid_df['FID']))
                kid_scores = {row['model']: (row['KID_mean'], row['KID_std'])
                              for _, row in fid_kid_df.iterrows()}
                plotter.plot_fid_kid(fid_scores, kid_scores)

            # --- Qualitative side-by-side grid (all available models) ---
            qualitative_samples = self._build_qualitative_samples(stage13 / "samples", df, limit=6)
            if qualitative_samples:
                plotter.plot_qualitative_grid(qualitative_samples, filename="qualitative_grid")
            else:
                self.logger.info("No Stage 15 sample folders found — skipping qualitative_grid.")

            # --- Per-sample metrics JSON (optional) ---
            # This JSON is a convenient single-file export: {sample_id: {model_name: {metric: value}}}.
            # The pipeline already produces most “distribution” plots from matrix_results.csv; this block
            # exists to ensure the JSON also has at least one explicit graph for debugging/presentations.
            per_sample_json = stage13 / "benchmarking_matrix" / "per_sample_metrics.json"
            if per_sample_json.exists():
                try:
                    with open(per_sample_json, "r", encoding="utf-8") as f:
                        payload = json.load(f)

                    rows = []
                    for sample_id, model_map in payload.items():
                        if not isinstance(model_map, dict):
                            continue
                        for model_name, metrics in model_map.items():
                            if not isinstance(metrics, dict):
                                continue
                            row = {"sample_id": sample_id, "model": model_name, "condition": "Unconditional"}
                            row.update(metrics)
                            rows.append(row)

                    if rows:
                        ps_df = pd.DataFrame(rows)
                        plotter.plot_metric_bars(ps_df, filename="per_sample_metric_bars")
                        plotter.plot_metric_distributions(ps_df, filename="per_sample_metric_distributions")
                except Exception as e:
                    self.logger.warning(f"Per-sample metrics JSON plotting failed: {e}")

            # --- Cross-validation ---
            cv_csv = stage13 / "statistical_tests" / "cross_validation_results.csv"
            if cv_csv.exists():
                cv_df = pd.read_csv(cv_csv)
                plotter.plot_cv_folds(cv_df)
                plotter.plot_cv_stability(cv_df)

            # --- V8 scientific completeness charts ---
            self._generate_v8_scientific_charts(
                plotter=plotter,
                stage06=stage06,
                stage12=stage12,
                stage13=stage13,
            )

            self.logger.info("All evaluation charts generated in Stage 17 output.")
        except Exception as e:
            self.logger.warning(f"Evaluation chart generation failed: {e}")

    def _generate_v8_scientific_charts(
        self,
        *,
        plotter: ThesisPlotter,
        stage06: Path,
        stage12: Path,
        stage13: Path,
    ):
        self.logger.info("Generating V8 scientific completeness charts…")

        # 1) Spatial contamination distribution
        regen_path = stage06 / "caption_spatial_regeneration_report.json"
        if regen_path.exists():
            try:
                regeneration_report = json.loads(regen_path.read_text(encoding="utf-8"))
                plotter.plot_spatial_contamination_distribution(regeneration_report)
            except Exception as e:
                self.logger.warning(f"Spatial contamination chart skipped: {e}")

        # 2) Stage06b grounding metrics panel
        grounding_path = stage06 / "stage_06b_grounding_validation.json"
        if grounding_path.exists():
            try:
                grounding_report = json.loads(grounding_path.read_text(encoding="utf-8"))
                plotter.plot_grounding_validation_panel(grounding_report)
            except Exception as e:
                self.logger.warning(f"Grounding validation panel skipped: {e}")

        # 3) Regime x source interaction grid
        regime_enabled = bool(
            self.config.get("model_training", {}).get("regime_comparison", {}).get("enabled", False)
        )
        interaction_df = self._load_regime_source_interaction(stage12)
        if interaction_df is not None and not interaction_df.empty:
            try:
                plotter.plot_regime_source_interaction(interaction_df)
            except Exception as e:
                self.logger.warning(f"Regime/source interaction chart skipped: {e}")
        elif not regime_enabled:
            self.logger.info("Regime/source interaction chart intentionally skipped (disabled in config).")
        else:
            self.logger.info("No regime/source interaction artifact found; chart skipped.")

        # 4) Frozen control integrity table
        frozen_df = self._load_frozen_control_integrity(stage13)
        if frozen_df is not None and not frozen_df.empty:
            try:
                plotter.plot_frozen_control_integrity_table(frozen_df)
            except Exception as e:
                self.logger.warning(f"Frozen control integrity table skipped: {e}")

        # 5) Expert reliability heatmap
        expect_expert_responses = bool(
            self.config.get("reporting", {}).get("optional_artifacts", {}).get("require_expert_responses", False)
        )
        expert_df, reliability_payload = self._load_expert_reliability_inputs()
        if expert_df is not None and not expert_df.empty:
            try:
                plotter.plot_expert_reliability_heatmap(expert_df, reliability_payload)
            except Exception as e:
                self.logger.warning(f"Expert reliability heatmap skipped: {e}")
        elif not expect_expert_responses:
            self.logger.info("Expert reliability heatmap intentionally skipped (responses not required at run time).")
        else:
            self.logger.info("No expert response artifact found; reliability heatmap skipped.")

    def _load_regime_source_interaction(self, stage12: Path) -> pd.DataFrame | None:
        csv_candidates = [
            stage12 / "interaction_analysis" / "regime_source_interaction.csv",
            stage12 / "interaction_analysis" / "regime_source_conditioning_delta.csv",
        ]
        for path in csv_candidates:
            if path.exists():
                return pd.read_csv(path)

        report_path = stage12 / "interaction_analysis" / "regime_comparison_report.json"
        if not report_path.exists():
            return None

        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows = []
        for regime_key, regime_label in (("biased_regime", "biased"), ("balanced_regime", "balanced")):
            payload = report.get(regime_key, {}) if isinstance(report.get(regime_key, {}), dict) else {}
            deltas = payload.get("conditioning_delta_by_source", {})
            if isinstance(deltas, dict) and deltas:
                for src in ("wikimedia", "europeana", "combined"):
                    if src in deltas:
                        rows.append({"regime": regime_label, "source_split": src, "delta_psnr": deltas.get(src)})
        if rows:
            return pd.DataFrame(rows)
        return None

    def _load_frozen_control_integrity(self, stage13: Path) -> pd.DataFrame | None:
        matrix_dir = stage13 / "benchmarking_matrix"
        detailed_path = matrix_dir / "frozen_control_integrity.json"
        if detailed_path.exists():
            payload = json.loads(detailed_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return pd.DataFrame(payload)
            if isinstance(payload, dict):
                rows = payload.get("rows")
                if isinstance(rows, list):
                    return pd.DataFrame(rows)

        manifest_path = matrix_dir / "frozen_control_manifest.json"
        if not manifest_path.exists():
            return None
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not payload:
            return None

        rows = []
        for model, sample_map in payload.items():
            if not isinstance(sample_map, dict):
                continue
            values = [str(v) for v in sample_map.values()]
            rows.append(
                {
                    "model": str(model),
                    "samples_hashed": int(len(sample_map)),
                    "unique_hashes": int(len(set(values))),
                    "requires_grad": "n/a",
                    "trainable_params": "n/a",
                    "status": "manifest_present",
                }
            )
        return pd.DataFrame(rows)

    def _load_expert_reliability_inputs(self) -> tuple[pd.DataFrame | None, dict | None]:
        project_root = Path(__file__).resolve().parents[3]
        stage10 = self.config_manager.get_stage_artifact_dir("S18")

        response_candidates = [
            project_root / "expert_validation_artifacts" / "athena_responses.csv",
            project_root / "expert_validation_artifacts" / "responses.csv",
            stage10 / "responses.csv",
        ]
        quality_candidates = [
            project_root / "expert_validation_artifacts" / "athena_quality_report.json",
            project_root / "expert_validation_artifacts" / "quality_report.json",
            stage10 / "quality_report.json",
        ]

        responses_path = next((p for p in response_candidates if p.exists()), None)
        if responses_path is None:
            return None, None

        df = pd.read_csv(responses_path)
        if "is_attention_check" in df.columns:
            df["is_attention_check"] = df["is_attention_check"].astype(str).str.lower().isin(("1", "true", "yes"))

        reliability_payload = None
        quality_path = next((p for p in quality_candidates if p.exists()), None)
        if quality_path is not None:
            try:
                quality_json = json.loads(quality_path.read_text(encoding="utf-8"))
                if isinstance(quality_json, dict) and isinstance(quality_json.get("reliability"), dict):
                    reliability_payload = quality_json.get("reliability")
                elif isinstance(quality_json, dict):
                    reliability_payload = quality_json
            except Exception:
                reliability_payload = None

        return df, reliability_payload

    def _build_qualitative_samples(self, samples_root: Path, matrix_df: pd.DataFrame, limit: int = 6):
        if not samples_root.exists():
            return []

        metrics_lookup = {}
        if {'sample_id', 'model', 'condition', 'psnr', 'ssim'}.issubset(matrix_df.columns):
            unconditional = matrix_df[matrix_df['condition'] == 'Unconditional']
            for _, row in unconditional.iterrows():
                stem = Path(str(row['sample_id'])).stem
                metrics_lookup.setdefault(stem, {})[str(row['model'])] = {
                    'psnr': float(row['psnr']) if pd.notna(row['psnr']) else np.nan,
                    'ssim': float(row['ssim']) if pd.notna(row['ssim']) else np.nan,
                }

        def _load_image(path: Path):
            if not path.exists():
                return None
            return np.array(Image.open(path).convert("RGB"))

        qualitative = []
        for sample_dir in sorted([p for p in samples_root.iterdir() if p.is_dir()]):
            sample = {
                'name': sample_dir.name,
                'original': _load_image(sample_dir / "original.png"),
                'masked_input': _load_image(sample_dir / "masked_input.png"),
                'metrics': metrics_lookup.get(sample_dir.name, {}),
            }
            for model_name, file_name in self.SAMPLE_MODEL_FILE_MAP.items():
                img = _load_image(sample_dir / file_name)
                if img is not None:
                    sample[model_name] = img

            if sample['original'] is not None and sample['masked_input'] is not None:
                qualitative.append(sample)
            if len(qualitative) >= limit:
                break

        return qualitative

if __name__ == '__main__':
    cm = ConfigManager(); stage = ReportingStage(cm); stage.run()



