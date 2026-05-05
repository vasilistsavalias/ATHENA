import sys
import subprocess
import os
import argparse
import json
from pathlib import Path
from datetime import datetime

def setup_environment(project_root):
    """
    Ensures the virtual environment is created and dependencies are installed.
    Returns the path to the python executable within the venv.
    """
    # Create venv in the project root to keep it central
    venv_dir = project_root / ".venv"
    
    if sys.platform == "win32":
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:
        venv_python = venv_dir / "bin" / "python"

    # 1. Create venv if it doesn't exist
    if not venv_dir.exists():
        print(f">>> Creating virtual environment at {venv_dir}...")
        try:
            subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
        except subprocess.CalledProcessError:
            print("ERROR: Failed to create virtual environment.")
            sys.exit(1)

    # 2. Install Dependencies
    # We check for a marker or just always attempt install (pip is fast if satisfied)
    print(">>> Checking/Installing dependencies...")
    requirements_path = project_root / "pipeline_requirements.txt"
    package_path = project_root
    
    try:
        # Upgrade pip first
        subprocess.check_call([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
        
        # Install requirements
        if requirements_path.exists():
            # Always allow upgrades so the venv doesn't get "stuck" on incompatible transitive versions
            subprocess.check_call(
                [str(venv_python), "-m", "pip", "install", "-r", str(requirements_path), "--upgrade", "--quiet"]
            )
        else:
            print(f"WARNING: pipeline_requirements.txt not found at {requirements_path}")

        # Install the package itself in editable mode (for src/ imports)
        subprocess.check_call([str(venv_python), "-m", "pip", "install", "-e", str(package_path), "--quiet"])

        # Enforce core scientific stack constraints (accelerate/torch compatibility on Linux/Windows).
        subprocess.check_call(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--force-reinstall",
                "numpy<2.0.0",
                "pandas<3.0.0",
                "Pillow<11.0.0",
                "--quiet",
            ]
        )

        # Guardrail: pip resolver can leave the venv in an incompatible state across machines.
        # Enforce a known-compatible HF stack after dependency resolution.
        subprocess.check_call(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--force-reinstall",
                "diffusers>=0.36.0,<0.37.0",
                "transformers>=4.26.0,<5.0.0",
                "accelerate>=0.16.0,<1.0.0",
                "huggingface_hub>=0.23.0,<1.0.0",
                "peft>=0.17.0,<1.0.0",
                "--quiet",
            ]
        )

        # Smoke import to fail fast during setup instead of mid-pipeline import.
        subprocess.check_call(
            [
                str(venv_python),
                "-c",
                "import numpy, pandas, PIL, diffusers, huggingface_hub, peft, transformers, accelerate; "
                "print(numpy.__version__, pandas.__version__, PIL.__version__, diffusers.__version__, "
                "huggingface_hub.__version__, peft.__version__, transformers.__version__, accelerate.__version__)",
            ],
            stdout=subprocess.DEVNULL,
        )
        
    except subprocess.CalledProcessError:
        print("ERROR: Failed to install dependencies.")
        sys.exit(1)

    return venv_python

def save_git_state(project_root):
    """Captures git commit hash and dirty status."""
    repro_dir = project_root / "outputs" / "00_reproducibility"
    repro_dir.mkdir(parents=True, exist_ok=True)
    git_state_file = repro_dir / "git_state.json"

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=project_root).decode().strip()
        is_dirty = bool(status)
        
        state = {
            "timestamp": datetime.now().isoformat(),
            "commit_hash": commit,
            "is_dirty": is_dirty,
            "status_output": status
        }
        
        with open(git_state_file, "w") as f:
            json.dump(state, f, indent=4)
            
        print(f">>> Git state saved to {git_state_file}")
        if is_dirty:
            print("WARNING: Repository has uncommitted changes. Results may not be reproducible.")
            
    except Exception as e:
        print(f"WARNING: Failed to capture git state: {e}")

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--setup-only", action="store_true")
    args, remaining_argv = parser.parse_known_args()

    # 1. Resolve Project Root
    project_root = Path(__file__).parent.resolve()
    
    # 2. Setup Environment (Auto-Magic)
    python_exec = setup_environment(project_root)
    if args.setup_only:
        print(">>> Setup-only mode: environment is ready.")
        return
    
    # 3. Resolve Script Paths
    main_script = project_root / "src" / "thesis_pipeline" / "main.py"
    preflight_script = project_root / "scripts" / "pipeline" / "preflight_check.py"
    checksum_script = project_root / "scripts" / "pipeline" / "generate_checksums.py"
    src_path = project_root / "src"
    
    # 5. Set Environment Variables
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_path) + os.pathsep + env.get("PYTHONPATH", "")

    # --- PHASE 1: PREFLIGHT CHECK ---
    print("\n>>> Running Pre-flight Check...")
    try:
        subprocess.check_call([str(python_exec), str(preflight_script)], cwd=project_root, env=env)
    except subprocess.CalledProcessError:
        print(">>> Pre-flight Check Failed. Aborting pipeline.")
        sys.exit(1)

    # --- PHASE 2: REPRODUCIBILITY LOGGING ---
    print("\n>>> Saving Git State...")
    save_git_state(project_root)

    # --- PHASE 3: EXECUTE PIPELINE ---
    cmd = [str(python_exec), str(main_script)] + remaining_argv
    
    print(f"\n>>> Running Pipeline with args: {remaining_argv}")
    print(f">>> Working Directory: {project_root}")
    
    pipeline_success = True
    try:
        subprocess.check_call(cmd, cwd=project_root, env=env)
    except subprocess.CalledProcessError as e:
        print(f"\n>>> Pipeline failed with exit code {e.returncode}")
        pipeline_success = False
    except KeyboardInterrupt:
        print("\n>>> Pipeline execution interrupted by user.")
        pipeline_success = False

    # --- PHASE 4: CHECKSUMS (Even if pipeline fails, we want to know what was produced) ---
    print("\n>>> Generating Artifact Checksums...")
    try:
        subprocess.check_call([str(python_exec), str(checksum_script)], cwd=project_root, env=env)
    except Exception as e:
        print(f"WARNING: Failed to generate checksums: {e}")

    if not pipeline_success:
        sys.exit(1)

if __name__ == "__main__":
    main()
