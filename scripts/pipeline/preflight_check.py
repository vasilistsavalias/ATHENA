import sys
import os
import shutil
import platform
import torch
import subprocess
import logging
from pathlib import Path
import psutil

# Configure logging
LOG_DIR = Path("outputs/00_reproducibility")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "preflight_check.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    filemode="w"  # Overwrite each time
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)

def log_info(msg):
    logging.info(msg)

def log_error(msg):
    logging.error(msg)

def check_python_version():
    log_info(f"Python Version: {sys.version}")
    if sys.version_info < (3, 8):
        log_error("FAIL: Python 3.8+ is required.")
        return False
    return True

def check_cuda():
    if not torch.cuda.is_available():
        log_error("WARNING: CUDA is not available. Training will be slow.")
        return True
    
    device_count = torch.cuda.device_count()
    log_info(f"CUDA Available: Yes")
    log_info(f"Device Count: {device_count}")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        log_info(f"GPU {i}: {props.name} | VRAM: {props.total_memory / 1e9:.2f} GB")
        
    return True

def check_nvidia_smi():
    try:
        # Just check if we can run it and get a version
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            log_info("nvidia-smi: OK")
            log_info(f"Driver Info: {result.stdout.splitlines()[0]}")
            return True
        else:
            if torch.cuda.is_available():
                log_info(f"WARNING: nvidia-smi returned error code {result.returncode}, but CUDA is available. Continuing...")
                return True
            log_error(f"FAIL: nvidia-smi returned error code {result.returncode}")
            return False
    except FileNotFoundError:
        log_error("WARNING: nvidia-smi not found. GPU monitoring unavailable.")
        return True

def _self_pid_set():
    current = psutil.Process(os.getpid())
    pids = {current.pid}
    try:
        for parent in current.parents():
            pids.add(parent.pid)
    except Exception:
        pass
    return pids

def check_gpu_occupancy():
    if not shutil.which("nvidia-smi"):
        log_info("GPU occupancy check skipped: nvidia-smi not available.")
        return True

    threshold_mb = int(os.environ.get("PIPELINE_GPU_OCCUPANCY_THRESHOLD_MB", "4096"))
    required_free_gpus = int(os.environ.get("PIPELINE_REQUIRED_FREE_GPUS", "1"))

    gpu_cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ]
    gpu_result = subprocess.run(gpu_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    free_gpus = []
    if gpu_result.returncode == 0:
        for raw in (gpu_result.stdout or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                gpu_idx = int(parts[0])
                used_mb = int(float(parts[1]))
            except Exception:
                continue
            if used_mb < threshold_mb:
                free_gpus.append(gpu_idx)

    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        log_info(
            f"GPU occupancy check skipped (nvidia-smi query failed with rc={result.returncode})."
        )
        return True

    offenders = []
    self_pids = _self_pid_set()
    for raw in (result.stdout or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            proc = parts[1]
            used_mb = int(float(parts[2]))
        except Exception:
            continue
        if pid in self_pids:
            continue
        if used_mb >= threshold_mb:
            offenders.append({"pid": pid, "process": proc, "used_mb": used_mb})

    if required_free_gpus <= 0:
        if offenders:
            pretty = "; ".join(
                f"pid={o['pid']} proc={o['process']} mem={o['used_mb']}MiB" for o in offenders
            )
            log_info(
                "GPU occupancy check bypassed (PIPELINE_REQUIRED_FREE_GPUS=0). "
                f"Observed heavy users: {pretty}"
            )
        else:
            log_info("GPU occupancy check bypassed (PIPELINE_REQUIRED_FREE_GPUS=0).")
        return True

    if len(free_gpus) < required_free_gpus:
        pretty = "; ".join(
            f"pid={o['pid']} proc={o['process']} mem={o['used_mb']}MiB" for o in offenders
        )
        log_error(
            "FAIL: Not enough free GPUs for pipeline start "
            f"(required={required_free_gpus}, free={len(free_gpus)}, threshold={threshold_mb}MiB). "
            f"Offenders: {pretty}"
        )
        return False

    if offenders:
        pretty = "; ".join(
            f"pid={o['pid']} proc={o['process']} mem={o['used_mb']}MiB" for o in offenders
        )
        log_info(
            f"GPU occupancy check passed with {len(free_gpus)} free GPUs "
            f"(required={required_free_gpus}, threshold={threshold_mb}MiB). "
            f"High-usage processes present on other GPUs: {pretty}"
        )
    else:
        log_info(
            f"GPU occupancy check passed ({len(free_gpus)} free GPUs, "
            f"required={required_free_gpus}, threshold={threshold_mb}MiB)."
        )
    return True

def check_disk_space(path=".", min_gb=10):
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (2**30)
    log_info(f"Disk Free Space ({path}): {free_gb:.2f} GB")
    
    if free_gb < min_gb:
        log_error(f"FAIL: Free space < {min_gb} GB. Risk of IO failure.")
        return False
    return True

def check_directories():
    print("--- Running check_directories ---")
    required_dirs = [
        "src",
        "config"
    ]
    all_ok = True
    for d in required_dirs:
        if os.path.isdir(d):
            print(f"Directory OK: {d}")
        else:
            print(f"FAIL: Critical directory missing: {d}")
            all_ok = False
    
    if all_ok:
        print("--- check_directories PASSED ---")
        return True
    else:
        print("--- check_directories FAILED ---")
        return False

def main():
    log_info("=== PREFLIGHT CHECK START ===")
    
    checks = [
        check_python_version,
        check_directories,
        check_disk_space,
        check_nvidia_smi,
        check_gpu_occupancy,
        check_cuda
    ]
    
    failed = False
    for check in checks:
        log_info(f"--- Running {check.__name__} ---")
        if not check():
            failed = True
            log_error(f"--- {check.__name__} FAILED ---")
        else:
            log_info(f"--- {check.__name__} PASSED ---")
            
    if failed:
        log_error("=== PREFLIGHT CHECK FAILED. FIX ISSUES BEFORE RUNNING. ===")
        sys.exit(1)
    else:
        log_info("=== PREFLIGHT CHECK PASSED. SYSTEM READY. ===")
        sys.exit(0)

if __name__ == "__main__":
    main()
