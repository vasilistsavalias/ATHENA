# Pipeline Workflow

Narrative execution guide for the 19-stage ATHENA pipeline. Each stage section explains what it does, why that approach was chosen over alternatives, and what artifacts it produces.

Canonical entry points:

- `src/thesis_pipeline/pipeline/registry.py` — stage registration
- `src/thesis_pipeline/pipeline/app.py` — orchestrator
- `scripts/pipeline/setup_and_run.sh` — operator wrapper

---

## Stage overview

```
S00 → S01 → S02 → S03 → S04 → S05 → S06
                                        ↓
S18 ← S17 ← S16 ← S15 ← S14 ← S13 ← S12 ← S11 ← S10 ← S09 ← S08 ← S07
```

| Stage | Name | Key technology |
| --- | --- | --- |
| S00 | Preflight | pytest (121 tests) |
| S01 | Research design | YAML config materialisation |
| S02 | Data acquisition | Wikimedia API + Europeana REST |
| S03 | Intelligent filtering | YOLOv8x |
| S04 | YOLO validation | Statistical distribution checks |
| S05 | Classical baselines | OpenCV inpainting |
| S06 | Exploratory data analysis | Distribution plots, bias audit |
| S07 | Caption generation | BLIP-2 |
| S08 | Caption refinement | Qwen2.5-7B |
| S09 | Data processing | Pillow resize to 512×512 PNG |
| S10 | Data splitting | Stratified 80/10/10 |
| S11 | Mask generation | Irregular / rectangular / edge masks |
| S12 | Hyperparameter tuning | Optuna (8-trial sweep) |
| S13 | Model training | Diffusers + Accelerate (K-fold, 4× A100) |
| S14 | Baseline fine-tuning | LaMa · MAT · CoModGAN |
| S15 | Full evaluation | PSNR · SSIM · LPIPS · FID · KID |
| S16 | Deployment artefacts | ONNX + PyTorch export |
| S17 | Reporting | Matplotlib plots, summary artefacts |
| S18 | Expert pack | `.zip` for evaluation platform |

---

## S00 — Preflight

**What:** Runs the full pytest suite (121 tests) as a gate before any compute stage executes. Aborts the pipeline if any test fails.

**Why preflight as a mandatory gate?** The pipeline takes 15–25 hours. A silent contract violation in stage S02 can propagate through filtering, captioning, training, and only surface at S15 evaluation — by which point 20+ hours of GPU time is wasted. Running tests first costs ~40 seconds and catches issues such as: wrong split ratios, stage registry mismatches, mask coverage guardrails, statistical test misconfiguration. This is a reproducibility discipline choice, not just a testing convenience.

**Why 121 tests?** Tests cover: unit correctness of every stage, statistical rigor checks (ensuring evaluation uses the right tests for the right distributions), governance preflight assertions (e.g. that the number of registered stages equals 19), and integration checks on the smoke-test fixtures.

---

## S01 — Research design

**What:** Materialises the YAML config into a reproducible run manifest. Records the git commit hash, config file path, and intended stage sequence. Writes `outputs/S01_research_design/run_manifest.json`.

**Why a dedicated stage for this?** Separating "what we intend to run" from "what we actually ran" creates a verifiable audit trail. The manifest is checked at the end to confirm that every intended stage ran.

---

## S02 — Data acquisition

**What:** Downloads images from two sources:
1. **Wikimedia Commons** — queried by category (Greek pottery, ancient vases, cultural heritage)
2. **Europeana** — queried by subject and type filters through their REST API

**Why Wikimedia + Europeana?**

- *Wikimedia* is well-indexed, freely licensed (CC), and has good coverage of well-known museum pieces. However it skews toward objects that were already famous before digitisation.
- *Europeana* aggregates institutional museum collections across Europe, providing less-photographed objects from smaller institutions. This diversity matters for generalisability: a model trained only on Wikimedia data may implicitly learn to reproduce compositions typical of flagship museum photographs.

**Why not scrape Google Images or other sources?** Licensing. Every image used in training and evaluation must have a verifiable open license. Both Wikimedia and Europeana provide license metadata per item, which is recorded in the acquisition manifest.

**Key configuration:**

```yaml
stage_02:
  wikimedia_target_count: 5000
  europeana_target_count: 5000
  europeana_api_key: ${EUROPEANA_API_KEY}
  strict_fail_policy: true  # fail hard if total count is zero
```

---

## S03 — Intelligent filtering

**What:** Runs YOLOv8x on every acquired image. Images are accepted if:
- At least one object of a target class (pottery, vessel, artifact) is detected
- The largest detected bounding box meets a minimum coverage threshold
- The image passes a bias audit (not flagged as a near-duplicate or uniformly-coloured image)

**Why YOLOv8x and not a simpler classifier?**

- *YOLOv8x* was chosen over a binary classifier because it provides bounding box localisation — downstream mask generation (S11) uses these boxes to place masks over artifact regions rather than arbitrary image areas.
- *YOLOv8x specifically (not YOLOv8s/m)* because the dataset contains small, detail-rich pottery fragments where smaller YOLO variants underperform. The extra compute cost is incurred only once at filtering.
- A simple ImageNet classifier was considered but rejected: it does not give us spatial extent, and fine-grained heritage object classes are poorly represented in standard classification benchmarks.

**Why a bias audit?**

- Without it, near-duplicate Wikimedia images of the same object from different angles would dominate the training split, inflating performance metrics. The bias audit uses perceptual hashing (pHash) to detect near-duplicates before splitting.

---

## S04 — YOLO validation

**What:** Statistical validation of the S03 filtering output. Checks that the distribution of detected object classes, bounding box coverage, and image source proportions meet pre-defined thresholds. Fails hard if distributions are abnormal.

**Why a separate validation stage?** Filtering errors (e.g. a bug in the coverage threshold) would silently pass through and corrupt training data. S04 is a contract-enforcement stage: it independently verifies S03's output rather than trusting the filter to self-report success.

---

## S05 — Classical baselines

**What:** Runs OpenCV's `INPAINT_TELEA` and `INPAINT_NS` algorithms on the filtered dataset using the same masks that will be used in the deep learning evaluation. Saves baseline metrics (PSNR, SSIM, LPIPS).

**Why compute classical baselines before training?** Classical baselines establish the minimum bar. A deep model that does not beat `INPAINT_TELEA` is not contributing anything meaningful. Computing them early means S15 evaluation has a fixed reference point regardless of how training goes.

**Why TELEA and NS specifically?** These are the two most-used classical inpainting algorithms in the academic literature. Including both ensures comparability with papers that report results on either method.

---

## S06 — Exploratory data analysis

**What:** Generates distribution plots and summary statistics over the accepted dataset: image dimensions, aspect ratios, colour histograms, bounding box coverage distribution, source proportions (Wikimedia vs Europeana), and class frequency.

**Why?** EDA at this stage (before captioning or processing) reveals skews that should be addressed in S09 processing or S10 splitting. For example, if 80% of images are portrait-oriented, the resize strategy in S09 must be adjusted to avoid significant content loss.

---

## S07 — Caption generation

**What:** Runs BLIP-2 (`Salesforce/blip2-opt-2.7b`) on every accepted image to generate a natural language description of the artifact's visual content.

**Why captions?** The fine-tuned Stable Diffusion model is conditioned on text prompts. Without high-quality captions describing each artifact, the model would need to be trained unconditioned — which eliminates the semantic guidance that separates fine-tuned diffusion from naive inpainting. Captions allow the model to understand "this is a red-figure kylix with a warrior scene" and use that context when reconstructing masked regions.

**Why BLIP-2 and not GPT-4V or CLIP?**

- *BLIP-2* is open-weight, runs locally, and produces fluent natural language descriptions without API costs. At 2.7B parameters it fits on a single A100.
- *GPT-4V* was considered but introduces API dependency, per-image cost, and breaks local reproducibility.
- *CLIP* does not generate text descriptions — it produces embeddings, which cannot be directly used as Stable Diffusion conditioning prompts.

---

## S08 — Caption refinement

**What:** Passes BLIP-2 captions through Qwen2.5-7B with a structured prompt that enriches them with: artifact type, estimated period, dominant colours, decorative motifs, visible damage, and contextual notes. Output is a refined caption optimised for SD inpainting conditioning.

**Why refinement?** Raw BLIP-2 captions often describe surface-level visual features ("a brown ceramic object") without domain-specific vocabulary that the SD text encoder has learned to associate with artistic styles. Qwen2.5-7B, being a strong instruction-following model, can rewrite captions in a more informative format given examples.

**Why Qwen2.5-7B and not Llama-3 or Mistral?**

- At 7B parameters it fits on a single A100 in fp16.
- Qwen2.5 series shows strong instruction-following on structured reformatting tasks compared to Mistral-7B at the same parameter count.
- Llama-3-8B was also evaluated; Qwen2.5-7B produced fewer hallucinated period attributions on validation samples.

---

## S09 — Data processing

**What:** Resizes every accepted image to 512×512 PNG using a centre-crop + resize strategy. Preserves aspect ratio by padding where needed.

**Why 512×512?** Stable Diffusion 1.x and 2.x are trained natively at 512×512. Using this resolution keeps the pre-trained spatial attention patterns intact. Higher resolutions (768×768, 1024×1024) would require tiled inference or fine-tuning at higher resolution, both of which substantially increase compute without a clear quality gain for the heritage artifact domain where fine detail is often already absent due to degradation.

**Why PNG?** Lossless format. Training images should not have JPEG compression artefacts as these can confuse the model about what constitutes authentic image texture vs compression noise.

---

## S10 — Data splitting

**What:** Splits the processed dataset into train (80%), validation (10%), test (10%) using stratified sampling by artifact class and source (Wikimedia/Europeana).

**Why stratified?** A random split risks having all Europeana images in the training set and none in the test set, making the test metrics non-representative of generalisation across sources. Stratification by source and class ensures proportional representation in all three splits.

**Why 80/10/10 and not 70/15/15?** With a typical dataset of 3,000–8,000 images, a 10% test set gives 300–800 test examples — sufficient for stable PSNR/SSIM/LPIPS estimates. A larger test split wastes training data without improving metric stability.

---

## S11 — Mask generation

**What:** Generates three types of inpainting masks for every image:

- **Irregular** — free-form polygonal masks of varying shape and coverage (10–60% of image area), simulating natural damage patterns
- **Rectangular** — axis-aligned boxes, simulating label damage and rectangular losses
- **Edge** — masks aligned to object boundaries from S03 YOLO boxes, simulating partial losses at artifact edges

**Why three mask types?** Different damage patterns in cultural heritage artifacts follow different spatial distributions. Evaluating on all three types reveals whether models handle structural edge damage differently from interior losses — which they do. A single mask type would hide mode-specific failures.

**Why irregular masks specifically for the "general damage" condition?** The literature on image inpainting (LaMa, MAT, CoModGAN papers) uses irregular masks following the convention of Liu et al. (2018). Matching this convention ensures fair comparison with baseline numbers from those papers.

---

## S12 — Hyperparameter tuning

**What:** Runs an 8-trial Optuna sweep over learning rate, batch size, gradient accumulation steps, and LoRA rank for the fine-tuned SD model.

**Why only 8 trials?** The compute budget constraint: an 8-trial sweep at ~2 epochs per trial on the full training set requires ~2–3 hours on 4× A100. Bayesian optimisation (Optuna's default TPE sampler) extracts more signal per trial than grid search, so 8 trials with TPE outperforms 16 trials with grid search for continuous hyperparameters.

**Why LoRA for fine-tuning?** Full fine-tuning of a 1.5B parameter diffusion model requires ~24 GB GPU RAM per process — at the limit of what 4× A100 can handle with a reasonable batch size. LoRA (Low-Rank Adaptation) reduces trainable parameters by ~100× (rank-4 adapters), enabling larger effective batch sizes and more stable training, while preserving the pre-trained model's semantic knowledge.

---

## S13 — Model training

**What:** Fine-tunes Stable Diffusion (SD 2.0 inpainting) with the best hyperparameters from S12, using K-fold cross-validation (K=5 by default), distributed across all available A100s via HuggingFace Accelerate.

**Why Stable Diffusion 2.0 inpainting and not SD 1.5 or DALL-E?**

- *SD 2.0 inpainting* was specifically pre-trained for masked image completion with both image and mask conditioning. SD 1.5 requires adapting a generation model to inpainting, which is less stable.
- *DALL-E / closed models* cannot be fine-tuned on custom data.
- *LaMa and MAT* (CNN/Transformer-based inpainters) are included as baselines (S14) but are not the primary model — diffusion models produce perceptually higher-quality completions on complex cultural textures.

**Why K-fold cross-validation?**

- With datasets in the 3,000–8,000 image range, a single train/val split produces metric estimates with high variance. K-fold gives 5 independent estimates of validation performance, enabling statistical confidence intervals on the final results.
- This is a reproducibility choice: the thesis compares models statistically (Wilcoxon signed-rank test), and valid statistical comparison requires multiple independent observations.

**Why Accelerate and not raw PyTorch DDP?**

- HuggingFace Accelerate abstracts multi-GPU setup (NCCL, device mapping, mixed precision) into a single `accelerate launch` call. This reduces setup code and makes it easy to change GPU count without modifying training scripts.
- `--mixed_precision=fp16` halves memory use and speeds up training ~1.5× on A100 tensor cores.

---

## S14 — Baseline fine-tuning

**What:** Fine-tunes LaMa, MAT, and CoModGAN on the same training split used for S13. Ensures baselines and the primary model are evaluated under identical conditions.

**Why fine-tune baselines and not use off-the-shelf weights?**

- Off-the-shelf LaMa/MAT weights were trained on Places2 (general scenes). Cultural heritage artifacts have different texture statistics (ceramic glazes, patina, fresco pigments). Fine-tuning on the same domain data gives baselines a fair chance and ensures the comparison reflects domain adaptation, not just pre-training data mismatch.

**Why LaMa, MAT, and CoModGAN specifically?**

- *LaMa* (Resolution-robust Large Mask inpainting): state-of-the-art CNN-based method, particularly strong on repetitive textures — relevant for patterned pottery.
- *MAT* (Mask-Aware Transformer): Transformer-based, better on large irregular masks where long-range context matters.
- *CoModGAN* (Co-Modulated GAN): GAN-based method representative of the previous generation of deep inpainters; included for historical comparison.

Together they cover the three major architecture paradigms in the literature (CNN, Transformer, GAN).

---

## S15 — Full evaluation

**What:** Evaluates all models (FT-SD, LaMa, MAT, CoModGAN) and classical baselines on the held-out test set across all three mask types. Reports:

- **PSNR** — pixel-level fidelity
- **SSIM** — structural similarity
- **LPIPS** — perceptual distance (AlexNet backbone)
- **FID** — distributional quality of completions vs. real images
- **KID** — FID variant with unbiased estimator for small test sets

Runs Wilcoxon signed-rank tests for all pairwise comparisons, with Bonferroni correction for multiple comparisons.

**Why these five metrics?** The metrics are complementary:

- PSNR/SSIM measure reference-based fidelity but do not capture perceptual quality well for creative completions.
- LPIPS is a learned perceptual metric that better correlates with human judgement than PSNR.
- FID/KID are reference-free distributional metrics that measure whether completions "look like" real images — important for diffusion models that may not reproduce the exact pixel values but produce plausible completions.

**Why Wilcoxon and not a t-test?**

- Metric scores (PSNR, SSIM, LPIPS) are not normally distributed — they have bounded ranges and skewed tails. Wilcoxon signed-rank test is non-parametric and does not assume normality.

---

## S16 — Deployment artefacts

**What:** Exports the best FT-SD model in two formats:
- **ONNX** — for deployment without a Python runtime
- **PyTorch `.pt`** — for further fine-tuning or inference with diffusers

---

## S17 — Reporting

**What:** Generates all evaluation plots and summary artefacts:

- Metric distribution plots (violin, bar, scatter)
- Per-mask-type breakdowns
- Training curve visualisations
- Hyperparameter sweep Pareto front
- Statistical significance matrices
- Qualitative sample grids (original / masked / completed)

**Why automated reporting as a pipeline stage?** Reproducibility. Every figure in the paper is generated deterministically from the same evaluation outputs. No manual screenshot or copy-paste is involved — re-running S17 regenerates every figure identically.

---

## S18 — Expert pack generation

**What:** Packages a stratified sample of evaluation triples (original image, masked image, model completions from all models) into a `.zip` archive with a JSON manifest (`campaign_manifest.json`) suitable for import into the ATHENA evaluation platform.

**What goes in the pack:**
- Block A items: single-model completion + original, for plausibility rating
- Block B items: pairwise comparisons between models
- Block C items: four-way model selection sets
- `campaign_manifest.json` describing all items, block assignments, and display metadata

This zip is imported directly into the platform admin dashboard and creates a campaign with all evaluation items pre-loaded.

---

## Artifact flow summary

```
Public sources (Wikimedia, Europeana)
  ↓ S02: acquisition
data/01_raw/combined_collection/
  ↓ S03: filtering
data/intermediate/02_filtered/accepted/
  ↓ S07+S08: captioning + refinement
outputs/S07_caption_generation/captions_enriched.json
  ↓ S09+S10: processing + splitting
data/intermediate/07_splits/{train,validation,test}/
  ↓ S11: mask generation
data/intermediate/08_inpainting/{train,validation,test}/
  ↓ S12: tuning → S13: training
data/intermediate/10_models/unet_best/
  ↓ S14: baseline fine-tuning
data/intermediate/10_models/{lama,mat,comodgan}_finetuned/
  ↓ S15: evaluation
outputs/S15_model_evaluation/benchmarking_matrix/matrix_results.csv
  ↓ S16+S17: deployment + reporting
outputs/S16_deployment_preparation/
outputs/S17_reporting/
  ↓ S18: expert pack
outputs/S18_expert_validation/campaign_pack.zip
```

---

## Failure discipline

- Trust `outputs/00_logs/run_state.json` over terminal scrollback.
- Resume from the **first failed stage**, never from the beginning unless doing a full reset.
- `--clean` / automatic artifact deletion was intentionally removed from the entrypoints. Use `clean_all_outputs.sh` explicitly for a full reset to keep deletion separated from orchestration.
- A stage that produces partial artifacts and crashes leaves those partial artifacts on disk. `--resume` will attempt to re-run the stage; if the stage is idempotent (most are), it will overwrite the partial artifacts correctly.

---

## See also

- [`../reference/artifacts.md`](../reference/artifacts.md) — complete artifact directory reference
- [`../reference/commands.md`](../reference/commands.md) — all flags and commands
- [`../reference/troubleshooting.md`](../reference/troubleshooting.md) — failure diagnostics
- [`../architecture/pipeline-architecture.md`](../architecture/pipeline-architecture.md) — runtime layers and execution path
