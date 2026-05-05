#!/bin/bash
set -euo pipefail

# Flags
FAST=0
SMOKE_TEST=0
RESUME=0
FORCE_RERUN_SUCCESSFUL=0
PHASE=""
CONFIG_OVERRIDE=""
STAGE02_LIMIT=""
RAW_DIR_OVERRIDE=""
ARTIFACTS_ROOT_OVERRIDE=""

ORIGINAL_ARGS=("$@")
i=1
while [ $i -le $# ]; do
    arg="${!i}"
    case "$arg" in
        --fast) FAST=1 ;;
        --smoke-test) SMOKE_TEST=1 ;;
        --resume) RESUME=1 ;;
        --force-rerun-successful) FORCE_RERUN_SUCCESSFUL=1 ;;
        --phase)
            i=$((i + 1))
            if [ $i -le $# ]; then
                PHASE="${!i}"
            else
                echo ">>> ERROR: --phase requires a value: integrity_200|full_test"
                exit 1
            fi
            ;;
        --config)
            i=$((i + 1))
            if [ $i -le $# ]; then
                CONFIG_OVERRIDE="${!i}"
            else
                echo ">>> ERROR: --config requires a value"
                exit 1
            fi
            ;;
        --stage02-limit)
            i=$((i + 1))
            if [ $i -le $# ]; then
                STAGE02_LIMIT="${!i}"
            else
                echo ">>> ERROR: --stage02-limit requires a value"
                exit 1
            fi
            ;;
        --raw-dir)
            i=$((i + 1))
            if [ $i -le $# ]; then
                RAW_DIR_OVERRIDE="${!i}"
            else
                echo ">>> ERROR: --raw-dir requires a value"
                exit 1
            fi
            ;;
        --artifacts-root)
            i=$((i + 1))
            if [ $i -le $# ]; then
                ARTIFACTS_ROOT_OVERRIDE="${!i}"
            else
                echo ">>> ERROR: --artifacts-root requires a value"
                exit 1
            fi
            ;;
    esac
    i=$((i + 1))
done

HAS_FULL_FLAG=0
for arg in "${ORIGINAL_ARGS[@]}"; do
    if [ "$arg" = "--full" ]; then
        HAS_FULL_FLAG=1
    fi
done

if [ "$SMOKE_TEST" -eq 1 ] && [ "$HAS_FULL_FLAG" -eq 0 ]; then
    HAS_FULL_FLAG=1
fi

# Directory resolution
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
RUN_SCRIPT="$PROJECT_ROOT/run.py"
MAIN_SCRIPT="$PROJECT_ROOT/src/thesis_pipeline/main.py"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
IOPAINT_REQS="$PROJECT_ROOT/scripts/pipeline/requirements_iopaint.txt"

POLICY_ARGS=()
if [ "$SMOKE_TEST" -eq 1 ]; then POLICY_ARGS+=("--smoke-test"); fi
if [ "$RESUME" -eq 1 ]; then POLICY_ARGS+=("--resume"); fi
if [ "$FORCE_RERUN_SUCCESSFUL" -eq 1 ]; then POLICY_ARGS+=("--force-rerun-successful"); fi
if [ -n "$PHASE" ]; then POLICY_ARGS+=("--phase" "$PHASE"); fi

FORWARD_ARGS=()
if [ -n "$CONFIG_OVERRIDE" ]; then FORWARD_ARGS+=("--config" "$CONFIG_OVERRIDE"); fi
if [ -n "$STAGE02_LIMIT" ]; then FORWARD_ARGS+=("--stage02-limit" "$STAGE02_LIMIT"); fi
if [ -n "$RAW_DIR_OVERRIDE" ]; then FORWARD_ARGS+=("--raw-dir" "$RAW_DIR_OVERRIDE"); fi
if [ -n "$ARTIFACTS_ROOT_OVERRIDE" ]; then FORWARD_ARGS+=("--artifacts-root" "$ARTIFACTS_ROOT_OVERRIDE"); fi

ensure_iopaint_hf_compat() {
    local py="$1"
    local hf_version="${2:-0.25.2}"
    "$py" -m pip install --upgrade --force-reinstall "huggingface_hub==$hf_version" >/dev/null
    "$py" - <<'PY'
import site
from pathlib import Path
import textwrap

p = Path(site.getsitepackages()[0]) / "sitecustomize.py"
p.write_text(
    textwrap.dedent(
        """
        import huggingface_hub as _hh
        try:
            from huggingface_hub.utils import is_offline_mode as _is_offline_mode
        except Exception:
            def _is_offline_mode():
                return False
        if not hasattr(_hh, "is_offline_mode"):
            _hh.is_offline_mode = _is_offline_mode
        if not hasattr(_hh, "cached_download"):
            try:
                from huggingface_hub import hf_hub_download as _hf_hub_download
                def cached_download(*args, **kwargs):
                    return _hf_hub_download(*args, **kwargs)
                _hh.cached_download = cached_download
            except Exception:
                pass
        """
    ).strip() + "\n",
    encoding="utf-8",
)
PY
}

ensure_iopaint_stack_health() {
    local venv_dir="$PROJECT_ROOT/.venv_iopaint"
    local iopaint_py="$venv_dir/bin/python"
    local iopaint_cli="$venv_dir/bin/iopaint"
    local verbose="${1:-0}"

    if [ ! -x "$iopaint_py" ] || [ ! -x "$iopaint_cli" ]; then
        echo ">>> ERROR: .venv_iopaint is missing/incomplete."
        return 1
    fi

    if ! "$iopaint_py" - <<'PY' >/dev/null 2>&1
from huggingface_hub import cached_download
import iopaint
import gdown
PY
    then
        echo ">>> Setup: Repairing .venv_iopaint huggingface compatibility..."
        ensure_iopaint_hf_compat "$iopaint_py" "0.25.2"
    fi

    if ! "$iopaint_py" - <<'PY' >/dev/null 2>&1
from huggingface_hub import cached_download
import iopaint
import gdown
PY
    then
        if [ "$verbose" -eq 1 ]; then
            "$iopaint_py" - <<'PY' || true
import traceback
try:
    import huggingface_hub as h
    from huggingface_hub import cached_download
    print("huggingface_hub:", h.__version__)
    print("has cached_download:", callable(cached_download))
    import iopaint
    import gdown
    print("iopaint/gdown import: OK")
except Exception:
    traceback.print_exc()
PY
        fi
        echo ">>> ERROR: IOPaint environment health check failed."
        return 1
    fi

    if ! command -v git >/dev/null 2>&1; then
        echo ">>> ERROR: git is required for CoModGAN/MI-GAN but was not found."
        return 1
    fi

    export IOPAINT_CLI="$iopaint_cli"
    return 0
}

ensure_iopaint_env() {
    if [ "${SKIP_IOPAINT_ENV:-0}" = "1" ]; then
        echo ">>> Setup: SKIP_IOPAINT_ENV=1 set; skipping IOPaint venv setup."
        return 0
    fi

    local venv_dir="$PROJECT_ROOT/.venv_iopaint"
    local iopaint_py="$venv_dir/bin/python"

    if [ -x "$venv_dir/bin/iopaint" ] && [ -x "$iopaint_py" ]; then
        if ensure_iopaint_stack_health 0; then
            return 0
        fi
        echo ">>> Setup: Existing .venv_iopaint is unhealthy. Rebuilding deterministically..."
    fi

    local python_bin="python3"
    echo ">>> Setup: Creating deterministic IOPaint venv at $venv_dir using $python_bin ..."
    rm -rf "$venv_dir"
    "$python_bin" -m venv "$venv_dir"
    "$iopaint_py" -m pip install -U pip setuptools wheel >/dev/null
    if [ -f "$IOPAINT_REQS" ]; then
        "$iopaint_py" -m pip install -r "$IOPAINT_REQS" >/dev/null
    else
        "$iopaint_py" -m pip install "iopaint==1.4.3" >/dev/null
    fi
    ensure_iopaint_hf_compat "$iopaint_py" "0.25.2"
    if ! ensure_iopaint_stack_health 1; then
        echo ">>> ERROR: IOPaint environment still unhealthy after rebuild."
        return 1
    fi
    return 0
}

ensure_apt_deps() {
    if ! command -v apt-get >/dev/null 2>&1; then
        return 0
    fi

    local pkgs=(build-essential python3-dev libjpeg-dev zlib1g-dev libpng-dev)
    echo ">>> Setup: Ensuring OS deps (Pillow/libjpeg) are installed: ${pkgs[*]}"

    if [ "$(id -u)" -eq 0 ]; then
        apt-get update
        apt-get install -y "${pkgs[@]}"
        return 0
    fi

    if command -v sudo >/dev/null 2>&1; then
        if sudo -n true 2>/dev/null; then
            sudo apt-get update
            sudo apt-get install -y "${pkgs[@]}"
            return 0
        fi
    fi

    echo ">>> Setup: WARNING: Could not auto-install OS deps (no passwordless sudo)."
    echo ">>> Setup: If Pillow build fails, run:"
    echo "    sudo apt-get update && sudo apt-get install -y ${pkgs[*]}"
    return 0
}

ensure_venv_ready() {
    echo ">>> Setup: Ensuring venv + Python deps (pip install -r requirements.txt)..."
    local setup_log="/tmp/thesis_pipeline_setup_only.log"
    python3 "$RUN_SCRIPT" --setup-only > "$setup_log" 2>&1 || {
        echo ">>> Setup: ERROR: setup-only failed. Last 120 log lines:"
        tail -n 120 "$setup_log" || true
        exit 1
    }

    if "$VENV_PYTHON" -c "import numpy, pandas, PIL, diffusers, huggingface_hub, transformers, accelerate; \
assert int(numpy.__version__.split('.')[0]) < 2; \
assert int(pandas.__version__.split('.')[0]) < 3; \
assert int(PIL.__version__.split('.')[0]) < 11; \
assert int(transformers.__version__.split('.')[0]) < 5; \
assert int(accelerate.__version__.split('.')[0]) < 1; \
assert int(huggingface_hub.__version__.split('.')[0]) < 1" >/dev/null 2>&1; then
        return 0
    fi

    echo ">>> Setup: Detected broken venv dependency state. Rebuilding .venv and retrying once..."
    rm -rf "$PROJECT_ROOT/.venv"
    python3 "$RUN_SCRIPT" --setup-only > /dev/null
}

# Helper: Check if a stage is in the requested args
has_stage() {
    local stage="$1"
    shift
    local args=("$@")
    local joined=" ${args[*]} "
    if [[ "$joined" == *" --full "* ]] || [[ "$joined" == *" --smoke-test "* ]]; then
        return 0
    fi
    if [[ "$joined" == *" $stage "* ]]; then
        return 0
    fi
    return 1
}

select_stage13_gpus() {
    local threshold_mb="${PIPELINE_GPU_OCCUPANCY_THRESHOLD_MB:-4096}"
    local requested="${PIPELINE_STAGE13_NUM_GPUS:-4}"
    select_free_gpus_for_phase "$requested" "$threshold_mb" "Stage S13" "PIPELINE_STAGE13"
}

select_free_gpus_for_phase() {
    local requested="$1"
    local threshold_mb="$2"
    local label="$3"
    local env_prefix="$4"

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo ">>> ERROR: nvidia-smi not found; cannot auto-select GPUs for ${label}."
        return 1
    fi

    local rows
    rows="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || true)"
    if [ -z "$rows" ]; then
        echo ">>> ERROR: Unable to query GPU memory usage for ${label} selection."
        return 1
    fi

    local free=()
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        local idx used
        idx="$(echo "$line" | cut -d',' -f1 | xargs)"
        used="$(echo "$line" | cut -d',' -f2 | xargs)"
        if [ -n "$idx" ] && [ -n "$used" ] && [ "$used" -lt "$threshold_mb" ]; then
            free+=("$idx")
        fi
    done <<< "$rows"

    local free_count="${#free[@]}"
    if [ "$free_count" -lt 1 ]; then
        echo ">>> ERROR: No free GPUs below threshold ${threshold_mb}MiB for ${label}."
        return 1
    fi

    local use_count="$requested"
    if [ "$free_count" -lt "$requested" ]; then
        use_count="$free_count"
        echo ">>> ${label}: only ${free_count} free GPU(s) found; downshifting to ${use_count}."
    fi
    if [ "$use_count" -lt 1 ]; then
        echo ">>> ERROR: Invalid selected GPU count: ${use_count}."
        return 1
    fi

    local selected=("${free[@]:0:$use_count}")
    local selected_csv
    selected_csv="$(IFS=,; echo "${selected[*]}")"
    local selected_gpus_var="${env_prefix}_SELECTED_GPUS"
    local selected_count_var="${env_prefix}_SELECTED_COUNT"
    export CUDA_VISIBLE_DEVICES="$selected_csv"
    export PIPELINE_REQUIRED_FREE_GPUS="$use_count"
    printf -v "$selected_gpus_var" '%s' "$selected_csv"
    printf -v "$selected_count_var" '%s' "$use_count"
    export "$selected_gpus_var" "$selected_count_var"
    echo ">>> ${label} GPU selection: CUDA_VISIBLE_DEVICES=${selected_csv} (threshold=${threshold_mb}MiB)."
    return 0
}

echo ">>> Wrapper: Analyzing execution strategy for args: ${ORIGINAL_ARGS[*]}"

PLAN_MSG=">>> Global Execution Plan:\n"
HAS_PRE=0
HAS_TRAIN=0
HAS_POST=0

# Canonical stage groups
# Preparation: S00-S12
STAGES_PRE="S00 S01 S02 S03 S04 S05 S06 S07 S08 S09 S10 S11 S12"
for s in $STAGES_PRE; do if has_stage "$s" "${ORIGINAL_ARGS[@]}"; then HAS_PRE=1; break; fi; done
# Training: S06 (legacy 12) — distributed with accelerate
if has_stage "S13" "${ORIGINAL_ARGS[@]}"; then HAS_TRAIN=1; fi
# Post-training: S14-S18
STAGES_POST="S14 S15 S16 S17 S18"
for s in $STAGES_POST; do if has_stage "$s" "${ORIGINAL_ARGS[@]}"; then HAS_POST=1; break; fi; done

if [ $HAS_PRE -eq 1 ]; then PLAN_MSG="$PLAN_MSG    [1/3] Preparation (S00-S12)  : Standard Python\n"; fi
if [ $HAS_TRAIN -eq 1 ]; then PLAN_MSG="$PLAN_MSG    [2/3] Training (S13)        : Accelerate Distributed (4x GPU)\n"; fi
if [ $HAS_POST -eq 1 ]; then PLAN_MSG="$PLAN_MSG    [3/3] Post-Training (S14-S18): Standard Python\n"; fi
echo -e "$PLAN_MSG"

# 1) Setup gates
if [ "$FAST" -eq 1 ]; then
    echo ">>> Fast mode enabled (--fast): skipping setup refresh + pytest gate."
    if [ ! -x "$VENV_PYTHON" ]; then
        echo ">>> ERROR: .venv missing in --fast mode. Run once without --fast first."
        exit 1
    fi
    ensure_iopaint_env
else
    ensure_apt_deps
    ensure_iopaint_env
    ensure_iopaint_stack_health
    ensure_venv_ready

    echo ">>> Verify: Running unit tests gate (pytest -q)..."
    "$VENV_PYTHON" -m pytest -q
fi

# 2) Execution logic
STAGES_TO_RUN_PRE=""
for s in $STAGES_PRE; do
    if has_stage "$s" "${ORIGINAL_ARGS[@]}"; then
        STAGES_TO_RUN_PRE="$STAGES_TO_RUN_PRE $s"
    fi
done

if [ -n "$STAGES_TO_RUN_PRE" ]; then
    echo ">>> Running Pre-Training Stages:$STAGES_TO_RUN_PRE"
    if has_stage "S03" "${ORIGINAL_ARGS[@]}" || has_stage "S07" "${ORIGINAL_ARGS[@]}" || has_stage "S08" "${ORIGINAL_ARGS[@]}"; then
        if ! select_free_gpus_for_phase "${PIPELINE_S03_NUM_GPUS:-4}" "${PIPELINE_GPU_OCCUPANCY_THRESHOLD_MB:-4096}" "Stage S03" "PIPELINE_S03"; then
            exit 1
        fi
    else
        export PIPELINE_REQUIRED_FREE_GPUS=0
        unset CUDA_VISIBLE_DEVICES PIPELINE_S03_SELECTED_GPUS PIPELINE_S03_SELECTED_COUNT
    fi
    "$VENV_PYTHON" "$RUN_SCRIPT" --stages $STAGES_TO_RUN_PRE ${POLICY_ARGS[@]+"${POLICY_ARGS[@]}"} ${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"}
fi

if has_stage "S13" "${ORIGINAL_ARGS[@]}"; then
    echo ">>> Running Training Stage: S13 (Distributed)"
    export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
    export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    if ! select_stage13_gpus; then
        exit 1
    fi

    echo ">>> Launching with Accelerate (Direct to main.py)..."
    ACCEL_ARGS=(--num_processes="${PIPELINE_STAGE13_SELECTED_COUNT}" --mixed_precision=fp16)
    if [ "${PIPELINE_STAGE13_SELECTED_COUNT}" -gt 1 ]; then
        ACCEL_ARGS=(--multi_gpu "${ACCEL_ARGS[@]}")
    fi
    "$VENV_PYTHON" -m accelerate.commands.launch \
        "${ACCEL_ARGS[@]}" \
        "$MAIN_SCRIPT" --stages S13 ${POLICY_ARGS[@]+"${POLICY_ARGS[@]}"} ${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"}
fi

STAGES_TO_RUN_POST=""
for s in $STAGES_POST; do
    if has_stage "$s" "${ORIGINAL_ARGS[@]}"; then
        STAGES_TO_RUN_POST="$STAGES_TO_RUN_POST $s"
    fi
done

if [ -n "$STAGES_TO_RUN_POST" ]; then
    echo ">>> Running Post-Training Stages:$STAGES_TO_RUN_POST"
    if has_stage "S15" "${ORIGINAL_ARGS[@]}"; then
        if ! select_free_gpus_for_phase "${PIPELINE_S15_NUM_GPUS:-1}" "${PIPELINE_GPU_OCCUPANCY_THRESHOLD_MB:-4096}" "Stage S15" "PIPELINE_S15"; then
            exit 1
        fi
    else
        export PIPELINE_REQUIRED_FREE_GPUS=0
        unset CUDA_VISIBLE_DEVICES PIPELINE_S15_SELECTED_GPUS PIPELINE_S15_SELECTED_COUNT
    fi
    "$VENV_PYTHON" "$RUN_SCRIPT" --stages $STAGES_TO_RUN_POST ${POLICY_ARGS[@]+"${POLICY_ARGS[@]}"} ${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"}
fi

echo ">>> Execution Complete."
