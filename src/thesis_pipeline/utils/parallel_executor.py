import torch
import multiprocessing
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Callable, Any
import math
import os

logger = logging.getLogger(__name__)

class ParallelExecutor:
    """
    Handles dynamic resource allocation and parallel execution 
    across CPUs and GPUs.
    """
    
    @staticmethod
    def get_device_map() -> List[str]:
        """
        Returns a list of available computation devices.
        e.g., ['cuda:0', 'cuda:1'] or ['cpu']
        """
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            logger.info(f"Detected {count} GPUs.")
            return [f"cuda:{i}" for i in range(count)]
        
        logger.info("No GPUs detected. Falling back to CPU.")
        return ["cpu"]

    @staticmethod
    def get_cpu_cores() -> int:
        """Returns the number of usable CPU cores."""
        # Leave 1 core free for system stability
        count = os.cpu_count() or 1
        return max(1, count - 1)

    @staticmethod
    def chunk_data(data: List[Any], num_chunks: int) -> List[List[Any]]:
        """Splits a list into N roughly equal chunks."""
        k, m = divmod(len(data), num_chunks)
        return [data[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(num_chunks)]

    @staticmethod
    def run_gpu_parallel(task_func: Callable, data: List[Any], **kwargs):
        """
        Runs a function in parallel across all available GPUs.
        
        Args:
            task_func: A function that accepts (device_id, subset_of_data, **kwargs)
            data: The full list of items to process.
        """
        devices = ParallelExecutor.get_device_map()
        num_devices = len(devices)
        
        if num_devices == 1 and devices[0] == "cpu":
            logger.warning("Running GPU task on CPU (Slow!).")
            return task_func("cpu", data, **kwargs)

        chunks = ParallelExecutor.chunk_data(data, num_devices)
        
        logger.info(f"Spawning {num_devices} processes for GPU parallelism...")
        
        ctx = multiprocessing.get_context('spawn')
        processes = []
        
        for i, device in enumerate(devices):
            # Pass the index 'i' as 'worker_id' so tqdm can use it for position
            p = ctx.Process(target=task_func, args=(device, chunks[i]), kwargs={**kwargs, 'worker_id': i})
            processes.append(p)
            p.start()
            
        for p in processes:
            p.join()
            if p.exitcode not in (0, None):
                raise RuntimeError(
                    f"Parallel GPU worker {p.pid} exited with code {p.exitcode} "
                    f"while running {getattr(task_func, '__name__', 'task')}"
                )
            
        logger.info("Parallel GPU execution complete.")

    @staticmethod
    def run_cpu_parallel(task_func: Callable, data: List[Any], use_threads=False, max_workers=None):
        """
        Runs a function in parallel across CPU cores.
        """
        workers = max_workers or ParallelExecutor.get_cpu_cores()
        Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        
        logger.info(f"Processing {len(data)} items with {workers} workers ({'Threads' if use_threads else 'Processes'})...")
        
        with Executor(max_workers=workers) as executor:
            # We assume task_func takes a single item
            results = list(executor.map(task_func, data))
            
        return results
