# GCP Pipeline Deployment Guide

This guide covers provisioning a GPU VM on Google Cloud, cloning ATHENA, running the full 19-stage pipeline end-to-end, and retrieving results. It assumes a fresh GCP project with billing enabled.

## Why GCP / why a GPU VM?

The pipeline spans CPU-bound I/O stages (data acquisition, filtering), mixed-precision GPU stages (YOLO inference, BLIP-2 captioning, Qwen2.5-7B refinement), and compute-intensive distributed training (S13 with Accelerate + 4× A100). A single powerful VM with multi-GPU access is simpler to manage than a Kubernetes cluster for a thesis-scale run. The `a2-highgpu-4g` machine type (4× NVIDIA A100-SXM4-40GB, 48 vCPUs, 340 GB RAM) is the smallest SKU that saturates S13 without memory-swapping. The pipeline auto-selects free GPUs at runtime via `nvidia-smi` — you do not need to configure `CUDA_VISIBLE_DEVICES` manually.

---

## Quick-reference commands

Once the VM is provisioned and the repo is cloned, these are the most-used commands:

```bash
# Connect
gcloud config set project YOUR_GCP_PROJECT_ID
gcloud compute ssh YOUR_USER@YOUR_VM_NAME --zone=us-central1-f

# On VM: go to repo
cd ~/athena

# Kill any stale tmux session and start fresh
tmux kill-session -t pipeline || true
tmux new -s pipeline

# Full run (Europeana key optional — pipeline continues without it)
EUROPEANA_API_KEY='YOUR_EUROPEANA_API_KEY' \
  bash scripts/pipeline/setup_and_run.sh --full

# Detach tmux without killing: Ctrl+B, then D
# Reattach later
tmux attach -t pipeline
```

---

## 1. One-time VM setup

### 1.1 Provision the VM

Recommended spec:

| Field | Value |
| --- | --- |
| Machine type | `a2-highgpu-4g` |
| vCPUs / RAM | 48 vCPUs, 340 GB |
| GPUs | 4× NVIDIA A100-SXM4-40GB |
| Boot disk | Ubuntu 20.04 LTS, 500 GB SSD |
| Zone | `us-central1-f` (has A100 quota) |

```bash
gcloud config set project YOUR_GCP_PROJECT_ID

gcloud compute instances create YOUR_VM_NAME \
  --zone=us-central1-f \
  --machine-type=a2-highgpu-4g \
  --accelerator=type=nvidia-tesla-a100,count=4 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=500GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --restart-on-failure
```

> **Why `--maintenance-policy=TERMINATE`?** GPU VMs cannot live-migrate. Without this flag GCP attempts migration during maintenance events and crashes the training run.

### 1.2 Install system packages

```bash
sudo apt-get update
sudo apt-get install -y git python3-venv zip tmux htop curl
```

### 1.3 Verify GPU access

```bash
nvidia-smi
# Expected: 4× A100 each showing ~40 GB memory
```

If `nvidia-smi` is not found, install CUDA drivers:

```bash
curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get install -y cuda-11-8
sudo reboot
```

### 1.4 Set up SSH deploy key for GitHub

```bash
# Generate key — no passphrase needed for CI/automation
ssh-keygen -t ed25519 -C "gcp-athena" -f ~/.ssh/id_ed25519 -N ""

# Print the public key to copy into GitHub
cat ~/.ssh/id_ed25519.pub
```

Add to your repo:

1. Go to `https://github.com/YOUR_ORG/athena/settings/keys`
2. **Add deploy key** → Title: `GCP VM` → paste the `ssh-ed25519 ...` line
3. Leave *Allow write access* unchecked (read-only is sufficient)

### 1.5 Clone the repository

```bash
git clone git@github.com:YOUR_ORG/athena.git ~/athena
cd ~/athena
chmod +x scripts/pipeline/setup_and_run.sh
chmod +x scripts/pipeline/clean_all_outputs.sh
chmod +x scripts/pipeline/dump_environment.sh
```

---

## 2. Running the pipeline

### 2.1 Always update code first

```bash
cd ~/athena
git fetch origin
git checkout main
git reset --hard origin/main
```

> **Why `git reset --hard`?** The pipeline writes `outputs/` and `.venv/` to the repo root, which can leave the working tree dirty. `reset --hard` clears any accidental tracked-file modifications without touching untracked directories.

### 2.2 Full pipeline run (recommended)

```bash
# Start inside a tmux session (see section 3)
EUROPEANA_API_KEY='YOUR_EUROPEANA_API_KEY' \
  ./scripts/pipeline/setup_and_run.sh --full
```

The script automatically:

- Creates `.venv` and `.venv_iopaint` if missing
- Runs `apt` dependency checks for Pillow build dependencies
- Runs the pytest gate — 121 tests must pass before any compute stage
- Auto-selects free GPUs via `nvidia-smi`
- Launches S13 via `accelerate launch --multi_gpu` if multiple free GPUs are found
- Resumes from run-state ledger when `--resume` is passed

### 2.3 Europeana-resilient mode

If you do not have a Europeana API key, or Europeana is temporarily unavailable:

```bash
cp config/pipeline/main_config.yaml config/pipeline/gcp_resilient.yaml

sed -i 's/strict_fail_policy: true/strict_fail_policy: false/' config/pipeline/gcp_resilient.yaml
sed -i 's/require_europeana_api_key: true/require_europeana_api_key: false/' config/pipeline/gcp_resilient.yaml
sed -i 's/require_enabled_source_nonzero: true/require_enabled_source_nonzero: false/' config/pipeline/gcp_resilient.yaml
sed -i 's/require_europeana_key_when_enabled: true/require_europeana_key_when_enabled: false/' config/pipeline/gcp_resilient.yaml

./scripts/pipeline/setup_and_run.sh --full \
  --config config/pipeline/gcp_resilient.yaml
```

> **Why two data sources?** Wikimedia Commons provides freely licensed images but skews toward well-known, frequently photographed artifacts. Europeana adds institutional collection coverage (museum digitisation programmes) with different provenance metadata. Using both reduces source bias in the evaluation dataset. The pipeline continues if Europeana is unavailable because Wikimedia alone still produces a valid dataset.

### 2.4 Smoke test

Runs the full 19-stage sequence on 50 images with 2 epochs and 2 K-fold splits. Completes in ~30 minutes on a GPU VM. Use this to verify the environment is healthy before committing to a 15–20 hour production run:

```bash
./scripts/pipeline/setup_and_run.sh --smoke-test
```

### 2.5 Resuming after interruption

The pipeline writes `outputs/00_logs/run_state.json` after every completed stage. Use `--resume` to skip already-completed stages:

```bash
./scripts/pipeline/setup_and_run.sh --resume --full
```

Force-restart from a specific stage:

```bash
./scripts/pipeline/setup_and_run.sh --resume --phase S13
```

Force-rerun stages that previously succeeded:

```bash
./scripts/pipeline/setup_and_run.sh --resume --force-rerun-successful --phase S12
```

### 2.6 Fast iteration (skip setup gate)

When `.venv` is already healthy and you are iterating on a single stage:

```bash
./scripts/pipeline/setup_and_run.sh --fast --phase S15
```

> **Warning:** `--fast` skips the pytest gate and venv refresh. Do not use on a fresh VM or after dependency changes.

### 2.7 Failure-recovery command matrix

| Scenario | Command |
| --- | --- |
| Resume full run after crash | `./setup_and_run.sh --resume --full` |
| Retry S12 sweep from checkpoint | `./setup_and_run.sh --resume --phase S12` |
| Resume S13 integrity phase | `./setup_and_run.sh --resume --phase integrity_200` |
| Run S13 final full-test phase | `./setup_and_run.sh --resume --phase full_test` |
| Continue downstream after S13 | `./setup_and_run.sh --resume --phase S14` |
| True fresh start | `./clean_all_outputs.sh && ./setup_and_run.sh --full` |

(All commands relative to `scripts/pipeline/`.)

---

## 3. Process management with tmux

Training (S13) takes 8–12 hours. **Always run inside a tmux session** so SSH disconnection does not kill the pipeline.

```bash
# Create session
tmux new -s pipeline

# Start pipeline inside session
./scripts/pipeline/setup_and_run.sh --full

# Detach without killing: Ctrl+B then D
# Reattach from any SSH session
tmux attach -t pipeline

# List sessions
tmux ls

# Kill stale session
tmux kill-session -t pipeline
```

### Monitoring the run

```bash
# Follow the main log in real-time
tail -f outputs/00_logs/thesis_pipeline*.log

# Check stage completion status
cat outputs/00_logs/run_state.json | python3 -m json.tool

# GPU utilisation live
watch -n 5 nvidia-smi
```

**Common tmux issue — "sessions should be nested with care":** you are already inside a tmux session (green status bar at bottom). Detach first with `Ctrl+B, D` rather than creating a new session.

---

## 4. Expected runtime

| Stage group | Duration | GPU % | Notes |
| --- | --- | --- | --- |
| S00–S02: Preflight + acquisition | 2–3 h | 0% | I/O-bound, downloads up to 10 K images |
| S03–S06: Filtering + EDA | 1–2 h | 10–30% | YOLOv8x inference on full dataset |
| S07–S08: Captioning + refinement | 2–4 h | 40–70% | BLIP-2 + Qwen2.5-7B per image |
| S09–S11: Processing + masks | 30 min | 5% | CPU resize + mask generation |
| S12: Hyperparameter tuning | 2–3 h | 80–95% | 8-trial Optuna sweep |
| S13: Model training | 8–12 h | 90–100% | Distributed 4× A100, K-fold |
| S14: Baseline fine-tuning | 2–4 h | 80–95% | LaMa + MAT + CoModGAN |
| S15: Evaluation | 1–2 h | 60–90% | Full benchmarking matrix |
| S16–S18: Deployment + pack | 15–30 min | 5% | Export, plots, zip |
| **Total** | **15–25 h** | — | Varies with K and dataset size |

---

## 5. Disk space management

A full run produces 50–100 GB of artifacts.

```bash
# Total project size
du -sh ~/athena

# Breakdown by top-level directory
du -h --max-depth=1 ~/athena | sort -hr

# Top 20 largest files
find ~/athena -type f -exec du -h {} + | sort -hr | head -20
```

Free space mid-run (safe to delete after respective stages complete):

```bash
# After S09 processing is done
rm -rf data/01_raw/wikimedia_collection/
rm -rf data/01_raw/europeana_collection/

# After S13 is done, remove intermediate checkpoints
find outputs/ -name "checkpoint-*" -type d -exec rm -rf {} +
```

> Never delete `data/intermediate/` or `outputs/` unless doing a full restart via `clean_all_outputs.sh`.

---

## 6. Retrieving results

### Package on the VM

```bash
cd ~/athena

# Lightweight (~200 MB, no model weights)
tar -czvf results_light.tar.gz \
  --exclude="*.safetensors" --exclude="*.bin" \
  --exclude="*.pth" --exclude="*.pt" \
  outputs/

# Full with models (~15 GB)
tar -czvf results_full.tar.gz outputs/

# Logs only (for debugging)
tar -czvf results_logs.tar.gz outputs/00_logs/

# Expert pack only (to import into evaluation platform)
tar -czvf expert_pack.tar.gz outputs/S18_expert_validation/
```

### Download to local machine

Run on your **local machine**, not inside the VM:

```bash
gcloud compute scp \
  YOUR_USER@YOUR_VM_NAME:~/athena/results_light.tar.gz \
  ~/Downloads/ \
  --zone=us-central1-f \
  --project YOUR_GCP_PROJECT_ID
```

Windows PowerShell:

```powershell
gcloud compute scp `
  YOUR_USER@YOUR_VM_NAME:~/athena/results_light.tar.gz `
  "$env:USERPROFILE\Downloads\" `
  --zone=us-central1-f `
  --project YOUR_GCP_PROJECT_ID
```

---

## 7. Post-run verification

```bash
# Verify all 19 stages completed
python3 - <<'PY'
import json, pathlib
state = json.loads(pathlib.Path("outputs/00_logs/run_state.json").read_text())
completed = set(state.get("completed_stages", []))
missing = [f"S{i:02d}" for i in range(19) if f"S{i:02d}" not in completed]
print("Completed:", sorted(completed))
print("Missing:", missing if missing else "none — all stages complete")
PY

# Key artifacts that must exist
ls -lh outputs/S15_model_evaluation/benchmarking_matrix/matrix_results.csv
ls -lh outputs/S18_expert_validation/
ls -lh outputs/S17_reporting/
```

---

## 8. Environment diagnostics

```bash
# Full environment dump (CUDA, Python, GPU, package versions)
./scripts/pipeline/dump_environment.sh

# Manual spot checks
nvidia-smi
python3 --version   # must be 3.9+
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
```

---

## 9. Troubleshooting

### `nvidia-smi: command not found`

CUDA drivers are not installed. Follow section 1.3.

### `No free GPUs below threshold`

All GPUs are occupied. Check and kill stale processes:

```bash
nvidia-smi
sudo kill -9 <PID>   # replace with the PID from nvidia-smi
```

### `setup-only failed` during venv setup

Delete the corrupted venv and retry:

```bash
rm -rf .venv
./scripts/pipeline/setup_and_run.sh --full
```

### IOPaint environment keeps failing health check

```bash
rm -rf .venv_iopaint
./scripts/pipeline/setup_and_run.sh --full
```

### S07/S08 OOM (captioning out of memory)

BLIP-2 and Qwen2.5-7B are memory-intensive. Check GPU memory with `nvidia-smi`. If the OOM is from the pipeline itself, reduce `batch_size` under `stage_07` and `stage_08` in `config/pipeline/main_config.yaml`.

### S13 training divergence / loss spikes

Check `outputs/S12_hyperparameter_tuning/best_hyperparameters.yaml`. If learning rate is above `3e-5`, re-run S12 with a tighter search range. Gradient explosion on cultural heritage imagery is common when learning rate is too high relative to the diffusion model's pre-trained weights.

### Divergent git history

```bash
git fetch origin
git checkout main
git reset --hard origin/main
```

### Pipeline appears stuck with no log output

```bash
tmux attach -t pipeline
tail -100 outputs/00_logs/thesis_pipeline*.log
```

If the log is not advancing, check for a background `pip install` that is waiting for user input (set `PIP_NO_INPUT=1` in the environment to prevent this).

---

## 10. Flags reference

| Flag | Effect |
| --- | --- |
| `--full` | Run all stages S00–S18 |
| `--smoke-test` | Use smoke-test config (50 images, 2 epochs) |
| `--resume` | Skip completed stages per `run_state.json` |
| `--phase S<N>` | Start from stage N |
| `--config <path>` | Use a custom YAML config |
| `--force-rerun-successful` | Re-run even stages marked complete |
| `--fast` | Skip setup refresh and pytest gate |
| `--stage02-limit <N>` | Cap data acquisition at N images |
| `--raw-dir <path>` | Override raw data root |
| `--artifacts-root <path>` | Override outputs root |

---

## Related docs

- [`config/pipeline/README.md`](../config/pipeline/README.md) — config schema and all YAML fields
- [`src/thesis_pipeline/stages/README.md`](../src/thesis_pipeline/stages/README.md) — per-stage contracts and expected artifacts
- [`src/thesis_pipeline/pipeline/README.md`](../src/thesis_pipeline/pipeline/README.md) — orchestration internals
- [`docs/reference/troubleshooting.md`](reference/troubleshooting.md) — extended failure diagnostics
- [`docs/reference/artifacts.md`](reference/artifacts.md) — complete artifact directory reference
- [`docs/reference/performance.md`](reference/performance.md) — cost drivers and tuning
