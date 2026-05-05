import hashlib
import os
from pathlib import Path

def generate_checksums(output_dir="outputs"):
    root = Path(output_dir)
    checksum_file = root / "00_reproducibility" / "checksums.sha256"
    
    # Ensure directory exists
    checksum_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating checksums for {root}...")
    
    with open(checksum_file, "w", encoding="utf-8", errors="replace") as f:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path != checksum_file:
                # Calculate hash
                sha256_hash = hashlib.sha256()
                try:
                    with open(path, "rb") as f_in:
                        for byte_block in iter(lambda: f_in.read(4096), b""):
                            sha256_hash.update(byte_block)
                    
                    # Write to file (relative path)
                    rel_path = path.relative_to(Path("."))
                    f.write(f"{sha256_hash.hexdigest()}  {rel_path}\n")
                except PermissionError:
                    print(f"Skipping {path} (PermissionError)")
                except Exception as e:
                    print(f"Error hashing {path}: {e}")
                    
    print(f"Checksums saved to {checksum_file}")

if __name__ == "__main__":
    generate_checksums()
