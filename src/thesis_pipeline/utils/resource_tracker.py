import time
import functools
import logging
import torch
import csv
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class ResourceTracker:
    """
    Decorator and utility to track computational resources (time, GPU memory).
    """
    def __init__(self, output_file: Path):
        self.output_file = output_file
        self._init_csv()

    def _init_csv(self):
        if not self.output_file.exists():
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'stage', 'duration_sec', 'gpu_mem_used_mb'])

    def track(self, stage_name: str):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                # Pre-run GPU check
                gpu_start = 0
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    gpu_start = torch.cuda.memory_allocated() / (1024 * 1024)

                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Stage {stage_name} failed: {e}")
                    raise
                finally:
                    duration = time.time() - start_time
                    gpu_end = 0
                    if torch.cuda.is_available():
                        gpu_end = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    
                    self.log_resource(stage_name, duration, gpu_end - gpu_start)
                
                return result
            return wrapper
        return decorator

    def log_resource(self, stage: str, duration: float, gpu_mem: float):
        with open(self.output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                stage,
                f"{duration:.2f}",
                f"{gpu_mem:.2f}"
            ])
        logger.info(f"Resource Usage [{stage}]: {duration:.2f}s, {gpu_mem:.2f}MB GPU")
