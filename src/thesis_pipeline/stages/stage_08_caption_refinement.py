from thesis_pipeline.logging_config import logger
from thesis_pipeline.config_manager import ConfigManager
from thesis_pipeline.utils.parallel_executor import ParallelExecutor
from thesis_pipeline.utils.stage_artifacts import resolve_stage_artifact_dir
from thesis_pipeline.stages.stage_07_caption_generation import _normalize_caption_text
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
import matplotlib.pyplot as plt
import re
import pandas as pd
import gc

from thesis_pipeline.components.prompt_utils import cap_prompt_to_token_budget

def _resolve_dataset_limit(config, stage_params):
    try:
        global_params = config.get("global_params", {}) if config else {}
        global_limit = global_params.get("dataset_limit")
    except Exception:
        global_limit = None

    try:
        stage_limit = stage_params.get("limit") if stage_params else None
    except Exception:
        stage_limit = None

    limits = [v for v in (global_limit, stage_limit) if isinstance(v, int) and v > 0]
    return min(limits) if limits else None


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)

def _refine_worker(device, caption_files, config_dict, worker_id=0):
    """
    Worker for Qwen Caption Refinement.
    """
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"[{device}] Loading Qwen: {model_name} (4-bit)")
    
    device_str = device if device != 'cpu' else 'cpu'
    
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Note: device_map="auto" might conflict with multiprocessing if not careful.
        # We manually move input tensors to 'device_str'.
        # However, 'load_in_4bit' usually requires device_map. 
        # For multi-GPU data parallel, we should force map to specific device.
        
        if device_str != 'cpu':
            device_map = {"": int(device_str.split(":")[1])}
        else:
            device_map = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True
        )
        model.eval()
    except Exception as e:
        print(f"[{device}] Model Load Error: {e}")
        report = {
            "worker_id": worker_id,
            "device": device,
            "model_name": model_name,
            "total_inputs": len(caption_files),
            "processed_count": 0,
            "failure_count": 0,
            "model_load_error": str(e),
            "aborted_due_to_fail_streak": True,
            "max_fail_streak_observed": 0,
            "failures": [],
        }
        (Path(config_dict['output_dir']) / f"worker_{worker_id}_refine_report.json").write_text(
            json.dumps(_json_safe(report), indent=2),
            encoding="utf-8",
        )
        return

    output_dir = Path(config_dict['output_dir'])
    max_new_tokens = int(config_dict.get("max_new_tokens", 120))
    max_worker_fail_streak = int(config_dict.get("max_worker_fail_streak", 10))
    report = {
        "worker_id": worker_id,
        "device": device,
        "model_name": model_name,
        "total_inputs": len(caption_files),
        "processed_count": 0,
        "failure_count": 0,
        "model_load_error": None,
        "aborted_due_to_fail_streak": False,
        "max_fail_streak_observed": 0,
        "failures": [],
    }
    
    processed = 0
    current_fail_streak = 0
    for cap_file in tqdm(caption_files, desc=f"[{device}] Refining", position=worker_id, leave=True):
        out_path = output_dir / cap_file.name
        
        if out_path.exists():
            report["processed_count"] += 1
            continue
            
        try:
            with open(cap_file, "r", encoding="utf-8") as f:
                raw_text = f.read()
                
            prompt = f"""<|im_start|>system
You are an expert archaeologist. Synthesize this raw data into a single, clean training caption.
<|im_end|>
<|im_start|>user
**Rules:**
1. Keep: Object type, style, figures, date.
2. Remove: URLs, Museum info, Copyright.
3. Output: Single paragraph, max 3 sentences, max ~60 words.
4. Use commas to list key attributes. No bullet points.

**Input:**
{raw_text}
<|im_end|>
<|im_start|>assistant
"""
            inputs = tokenizer(prompt, return_tensors="pt").to(device_str)
            
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            output_text = tokenizer.decode(generated_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

            # Best-effort formatting cleanup: collapse whitespace, remove accidental multi-paragraph output.
            output_text = re.sub(r"\s+", " ", output_text).strip()
            output_text = output_text.replace("\n", " ").strip()
            output_text = _normalize_caption_text(output_text)
            if not output_text:
                raise RuntimeError("refined caption normalized to empty text")
            
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(output_text)
            processed += 1
            report["processed_count"] += 1
            current_fail_streak = 0
            
        except Exception as e:
            print(f"[{device}] Error: {e}")
            report["failure_count"] += 1
            current_fail_streak += 1
            report["max_fail_streak_observed"] = max(
                int(report["max_fail_streak_observed"]),
                current_fail_streak,
            )
            report["failures"].append(
                {
                    "caption_file": cap_file.name,
                    "message": str(e),
                }
            )
            try:
                gc.collect()
                if device != "cpu" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            if current_fail_streak >= max_worker_fail_streak:
                report["aborted_due_to_fail_streak"] = True
                break

    print(f"[{device}] Done. Processed {processed}")
    (output_dir / f"worker_{worker_id}_refine_report.json").write_text(
        json.dumps(_json_safe(report), indent=2),
        encoding="utf-8",
    )

class CaptionRefinementStage:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.config = config_manager.config
        self.logger = logger
        # Primary: look for captions saved alongside filtered images
        self.input_dir = Path(self.config.paths.data.filtered) / "accepted" / "captions"
        # Fallback: look in the Stage 06 artifacts directory (where captions_raw.json lives)
        self.input_dir_fallback = resolve_stage_artifact_dir(self.config_manager, "S07")
        self.output_dir = Path(self.config.paths.data.filtered) / "accepted" / "refined_captions"
        self.artifacts_dir = resolve_stage_artifact_dir(self.config_manager, "S08")
        self.cr_cfg = self.config.get("caption_refinement", {}) if self.config else {}

    def _load_stage06_report(self) -> dict:
        report_path = self.input_dir_fallback / "caption_generation_report.json"
        if report_path.exists():
            return json.loads(report_path.read_text(encoding="utf-8"))
        return {}

    def _load_worker_reports(self) -> list[dict]:
        reports = []
        for report_file in sorted(self.output_dir.glob("worker_*_refine_report.json"), key=lambda p: p.name):
            reports.append(json.loads(report_file.read_text(encoding="utf-8")))
            report_file.unlink()
        return reports

    def _write_refinement_report(self, caption_files, worker_reports, stage06_report):
        refined_files = sorted(self.output_dir.glob("*.txt"), key=lambda p: p.name)
        input_caption_count = len(caption_files)
        refined_caption_count = len(refined_files)
        worker_failures = []
        model_load_errors = []
        max_fail_streak = 0
        aborted_workers = []
        for report in worker_reports:
            worker_failures.extend(report.get("failures", []))
            max_fail_streak = max(max_fail_streak, int(report.get("max_fail_streak_observed", 0) or 0))
            if report.get("model_load_error"):
                model_load_errors.append(report.get("model_load_error"))
            if report.get("aborted_due_to_fail_streak"):
                aborted_workers.append(report.get("worker_id"))

        base_total = int(stage06_report.get("input_image_count", input_caption_count) or input_caption_count)
        stage06_success_rate = float(stage06_report.get("success_rate", 1.0 if input_caption_count else 0.0) or 0.0)
        refinement_success_rate = (refined_caption_count / input_caption_count) if input_caption_count else 0.0
        report = {
            "stage06_input_image_count": base_total,
            "stage06_success_rate": stage06_success_rate,
            "input_caption_count": input_caption_count,
            "refined_caption_count": refined_caption_count,
            "missing_or_failed_count": max(0, input_caption_count - refined_caption_count),
            "refinement_success_rate": refinement_success_rate,
            "worker_reports": worker_reports,
            "worker_failures": worker_failures,
            "model_load_errors": model_load_errors,
            "aborted_workers": aborted_workers,
            "max_fail_streak_observed": max_fail_streak,
        }
        (self.artifacts_dir / "caption_refinement_report.json").write_text(
            json.dumps(_json_safe(report), indent=2),
            encoding="utf-8",
        )

        failure_policy = str(self.cr_cfg.get("failure_policy", "hard_fail")).lower()
        success_rate_min = float(self.cr_cfg.get("success_rate_min", 0.90) or 0.90)
        reasons = []
        if stage06_report and stage06_success_rate < success_rate_min:
            reasons.append(
                f"stage06 success rate {stage06_success_rate:.3f} below threshold {success_rate_min:.3f}"
            )
        if refinement_success_rate < success_rate_min:
            reasons.append(
                f"refinement success rate {refinement_success_rate:.3f} below threshold {success_rate_min:.3f}"
            )
        if model_load_errors:
            reasons.append("refinement worker model load failure detected")
        if aborted_workers:
            reasons.append(f"workers aborted due to fail streak: {aborted_workers}")
        if reasons and failure_policy == "hard_fail":
            raise RuntimeError("Caption refinement failed strict policy: " + " | ".join(reasons))

    def run(self):
        self.logger.info("=" * 20 + " STAGE 07: Caption Refinement (Parallel) " + "=" * 20)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.input_dir.exists():
            # Try fallback path
            if self.input_dir_fallback.exists():
                self.logger.info(f"Primary caption dir not found at {self.input_dir}. "
                                 f"Using fallback at {self.input_dir_fallback}")
                self.input_dir = self.input_dir_fallback
            else:
                self.logger.error(f"Input directory not found at {self.input_dir} or {self.input_dir_fallback}.")
                (self.artifacts_dir / "README.md").write_text(
                    "Caption refinement skipped: input captions directory not found.\n",
                    encoding="utf-8"
                )
                return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        stage06_report = self._load_stage06_report()
        
        caption_files = list(self.input_dir.glob("*.txt"))
        caption_files = sorted(caption_files, key=lambda p: p.name)
        self.logger.info(f"Found {len(caption_files)} captions.")
        if not caption_files:
            self._write_refinement_report([], [], stage06_report)
            self.logger.warning("Caption refinement skipped: no input caption files were available.")
            return

        limit = _resolve_dataset_limit(self.config, self.config.get("caption_refinement", {}))
        if limit:
            caption_files = caption_files[:limit]
            self.logger.info(f"Applying dataset limit: {limit} captions.")
        
        # --- Mock Mode ---
        if self.cr_cfg.get("mock", False):
            self.logger.info("MOCK MODE ENABLED: Skipping Qwen refinement.")
            processed = 0
            for cap_file in caption_files:
                out_path = self.output_dir / cap_file.name
                if not out_path.exists():
                    with open(cap_file, "r", encoding="utf-8") as f:
                        raw = f.read()
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(f"REFINED: {raw[:100]}...")
                    processed += 1
            self.logger.info(f"Mocked refinement: Created {processed} files.")
            # Even in mock mode, generate the same downstream artifacts so tests/docs stay consistent.
            mock_report = {
                "worker_id": 0,
                "device": "mock",
                "model_name": "mock",
                "total_inputs": len(caption_files),
                "processed_count": processed,
                "failure_count": 0,
                "model_load_error": None,
                "aborted_due_to_fail_streak": False,
                "max_fail_streak_observed": 0,
                "failures": [],
            }
            (self.output_dir / "worker_0_refine_report.json").write_text(
                json.dumps(mock_report, indent=2),
                encoding="utf-8",
            )
            worker_reports = self._load_worker_reports()
            self._write_refinement_report(caption_files, worker_reports, stage06_report)
            self._write_refined_caption_artifacts()
            self.logger.info("Stage S08 Completed (Mocked).")
            return

        worker_config = {
            "output_dir": str(self.output_dir),
            "max_new_tokens": int(self.cr_cfg.get("max_tokens", 120) or 120),
            "max_worker_fail_streak": int(self.cr_cfg.get("max_worker_fail_streak", 10) or 10),
        }
        
        ParallelExecutor.run_gpu_parallel(_refine_worker, caption_files, config_dict=worker_config)
        worker_reports = self._load_worker_reports()
        self._write_refinement_report(caption_files, worker_reports, stage06_report)

        # Create clip-safe refined captions + story artifacts (and overwrite downstream captions to clip-safe).
        self._write_refined_caption_artifacts()

        # Copy final (clip-safe) refined captions into artifacts for easy inspection.
        for cap_file in self.output_dir.glob("*.txt"):
            try:
                content = cap_file.read_text(encoding="utf-8")
                if not isinstance(content, str):
                    content = str(content)
                (self.artifacts_dir / cap_file.name).write_text(content, encoding="utf-8")
            except Exception:
                continue
        
        # Generate period coverage progression chart
        self._generate_period_coverage_chart()
        
        self.logger.info("Stage S08 Completed.")

    def _write_refined_caption_artifacts(self):
        """Build refined caption maps + clip-safe variants + prompt-story artifacts."""
        try:
            # Load Stage 06 captions for story plots
            stage_06_dir = resolve_stage_artifact_dir(self.config_manager, "S07")
            raw_path = stage_06_dir / "captions_raw.json"
            blip_path = stage_06_dir / "captions_enriched.json"
            raw_caps = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else {}
            blip_caps = json.loads(blip_path.read_text(encoding="utf-8")) if blip_path.exists() else {}

            # Build maps keyed like Stage 06: "<stem>.jpg"
            refined_raw: dict[str, str] = {}
            refined_clip: dict[str, str] = {}

            clip_max_tokens = 77
            # Word-budget guard: keep comfortably under 77 CLIP tokens without requiring tokenizer downloads.
            clip_safe_word_cap = int(self.cr_cfg.get("clip_safe_word_cap", 60) or 60)

            for txt_path in sorted(self.output_dir.glob("*.txt"), key=lambda p: p.name):
                try:
                    original = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
                except Exception:
                    continue

                key = f"{txt_path.stem}.jpg"
                refined_raw[key] = original

                capped, _ = cap_prompt_to_token_budget(
                    original,
                    tokenizer=None,
                    max_tokens=clip_safe_word_cap,
                )
                # Ensure 1 paragraph / no newlines for SD prompts.
                capped = re.sub(r"\s+", " ", capped or "").strip()
                refined_clip[key] = capped

                # Overwrite downstream caption file with clip-safe text (used by Stage 08β†’10 training).
                try:
                    txt_path.write_text(capped, encoding="utf-8")
                except Exception:
                    pass

            # Write JSON artifacts for Stage 13 prompt conditions
            (self.artifacts_dir / "refined_captions.json").write_text(
                json.dumps(refined_raw, indent=2),
                encoding="utf-8",
            )
            (self.artifacts_dir / "refined_captions_clip_safe.json").write_text(
                json.dumps(refined_clip, indent=2),
                encoding="utf-8",
            )

            # Prompt length progression (word counts)
            def _lens(d: dict) -> list[int]:
                return [len(str(v).split()) for v in (d or {}).values() if v is not None]

            rows = []
            for name, d in [
                ("metadata_raw", raw_caps),
                ("blip2_enriched", blip_caps),
                ("qwen_refined_raw", refined_raw),
                ("qwen_refined_clip_safe", refined_clip),
            ]:
                for v in _lens(d):
                    rows.append({"stage": name, "words": int(v)})
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(self.artifacts_dir / "prompt_length_progression.csv", index=False)
                try:
                    fig, ax = plt.subplots(figsize=(10, 4), dpi=250)
                    order = [
                        "metadata_raw",
                        "blip2_enriched",
                        "qwen_refined_raw",
                        "qwen_refined_clip_safe",
                    ]
                    data = [df[df["stage"] == s]["words"].values for s in order]
                    ax.boxplot(data, labels=[s.replace("_", "\n") for s in order], showfliers=False)
                    ax.set_ylabel("Words (proxy for CLIP tokens)")
                    ax.set_title("Prompt Length Progression (Metadata β†’ BLIP2 β†’ Qwen β†’ Clip-safe)")
                    ax.grid(axis="y", alpha=0.25)
                    plt.tight_layout()
                    plt.savefig(self.artifacts_dir / "prompt_length_progression.png", bbox_inches="tight")
                    plt.close()
                except Exception:
                    pass

            # Single example trace (first shared key across all stages)
            shared = None
            for k in sorted(set(raw_caps) & set(blip_caps) & set(refined_raw) & set(refined_clip)):
                shared = k
                break
            if shared:
                trace = [
                    "# Prompt Trace Example",
                    "",
                    f"Sample key: `{shared}`",
                    "",
                    "## Metadata (raw)",
                    raw_caps.get(shared, ""),
                    "",
                    "## BLIP2 (enriched)",
                    blip_caps.get(shared, ""),
                    "",
                    "## Qwen (refined, raw)",
                    refined_raw.get(shared, ""),
                    "",
                    "## Qwen (refined, clip-safe)",
                    refined_clip.get(shared, ""),
                    "",
                    f"Note: Clip budget target = {clip_max_tokens} tokens (SD 1.x CLIP).",
                ]
                (self.artifacts_dir / "prompt_trace_example.md").write_text("\n".join(trace), encoding="utf-8")
        except Exception as e:
            self.logger.warning(f"Failed to write refined caption artifacts: {e}")
    
    def _generate_period_coverage_chart(self):
        """Generate period coverage progression chart (Raw β†’ BLIP2 β†’ Qwen)."""
        try:
            self.logger.info("Generating period coverage progression chart...")
            
            # Period terminology
            period_terms = [
                'geometric', 'archaic', 'classical', 'hellenistic',
                'red-figure', 'black-figure', 'white-ground',
                'attic', 'corinthian', 'period', 'century', 'b.c.', 'bc'
            ]
            
            # Load raw captions (museum metadata)
            stage_06_dir = resolve_stage_artifact_dir(self.config_manager, "S07")
            raw_path = stage_06_dir / "captions_raw.json"
            blip2_path = stage_06_dir / "captions_enriched.json"
            
            if not raw_path.exists() or not blip2_path.exists():
                self.logger.warning("Stage 06 caption files not found. Skipping period coverage chart.")
                return
            
            with open(raw_path, 'r', encoding='utf-8') as f:
                raw_captions = json.load(f)
            
            with open(blip2_path, 'r', encoding='utf-8') as f:
                blip2_captions = json.load(f)
            
            # Count period coverage in each stage
            raw_matches = sum(
                1 for c in raw_captions.values()
                if any(term in str(c).lower() for term in period_terms)
            )
            raw_total = len(raw_captions)
            raw_pct = (raw_matches / raw_total * 100) if raw_total > 0 else 0
            
            blip2_matches = sum(
                1 for c in blip2_captions.values()
                if any(term in str(c).lower() for term in period_terms)
            )
            blip2_total = len(blip2_captions)
            blip2_pct = (blip2_matches / blip2_total * 100) if blip2_total > 0 else 0
            
            # Count Qwen-refined captions
            qwen_caption_files = list(self.output_dir.glob('*.txt'))
            qwen_matches = 0
            for cap_file in qwen_caption_files:
                try:
                    text = cap_file.read_text(encoding='utf-8', errors='ignore').lower()
                    if any(term in text for term in period_terms):
                        qwen_matches += 1
                except:
                    pass
            qwen_total = len(qwen_caption_files)
            qwen_pct = (qwen_matches / qwen_total * 100) if qwen_total > 0 else 0
            
            self.logger.info(f"Period coverage: Raw={raw_pct:.1f}%, BLIP2={blip2_pct:.1f}%, Qwen={qwen_pct:.1f}%")
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
            
            stages = ['Raw Metadata\\n(Museum Records)', 'BLIP2 Enriched\\n(Vision-Only)', 'Qwen-Refined\\n(Vision + Metadata)']
            percentages = [raw_pct, blip2_pct, qwen_pct]
            colors = ['#7E7E7E', '#E8A628', '#2E7D32']  # Gray, Orange, Green
            
            bars = ax.bar(stages, percentages, color=colors, edgecolor='black', linewidth=1.5)
            
            # Add percentage labels on bars
            for bar, pct, count, total in zip(bars, percentages,
                                               [raw_matches, blip2_matches, qwen_matches],
                                               [raw_total, blip2_total, qwen_total]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                        f'{pct:.1f}%\\n({count:,}/{total:,})',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            # Styling
            ax.set_ylabel('Period/Style Coverage (%)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Caption Pipeline Stage', fontsize=12, fontweight='bold')
            ax.set_title('Period Coverage Progression Through Caption Pipeline',
                         fontsize=14, fontweight='bold', pad=20)
            ax.set_ylim(0, 105)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            # Add annotation explaining the drop at BLIP2
            if blip2_pct < raw_pct:
                ax.annotate("Vision models can't date pottery\nfrom images alone",
                            xy=(1, blip2_pct), xytext=(1, 30),
                            arrowprops=dict(arrowstyle='->', color='red', lw=2),
                            fontsize=9, color='red', ha='center',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', alpha=0.8))
            
            # Add annotation explaining Qwen's recovery
            if qwen_pct > blip2_pct:
                ax.annotate('Qwen synthesizes vision\\n+ metadata context',
                            xy=(2, qwen_pct), xytext=(2, 75),
                            arrowprops=dict(arrowstyle='->', color='green', lw=2),
                            fontsize=9, color='green', ha='center',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='green', alpha=0.8))
            
            plt.tight_layout()
            
            # Save to artifacts
            output_path = self.artifacts_dir / "period_coverage_progression.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Period coverage chart saved to {output_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to generate period coverage chart: {e}")

if __name__ == "__main__":
    cm = ConfigManager()
    stage = CaptionRefinementStage(cm)
    stage.run()


