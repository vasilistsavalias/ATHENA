import pytest
from unittest.mock import patch, MagicMock
import os
import sys

if os.environ.get("TORCH_IMPORT_OK", "1") != "1":
    pytest.skip(
        "Skipping torch-dependent tests because torch import failed/crashed in a subprocess probe. "
        "Fix your torch install or set TORCH_IMPORT_OK=1 to force-run.",
        allow_module_level=True,
    )

pytest.importorskip("torch")
from thesis_pipeline.utils.parallel_executor import ParallelExecutor

def test_chunk_data():
    data = list(range(10))
    chunks = ParallelExecutor.chunk_data(data, 3)
    assert len(chunks) == 3
    # 10 / 3 -> 4, 3, 3 distribution
    assert len(chunks[0]) == 4
    assert len(chunks[1]) == 3
    assert len(chunks[2]) == 3
    assert sum(len(c) for c in chunks) == 10

def test_chunk_data_empty():
    chunks = ParallelExecutor.chunk_data([], 3)
    assert len(chunks) == 3
    assert all(len(c) == 0 for c in chunks)

@patch("torch.cuda.is_available")
@patch("torch.cuda.device_count")
def test_get_device_map_gpu(mock_count, mock_is_avail):
    mock_is_avail.return_value = True
    mock_count.return_value = 2
    devices = ParallelExecutor.get_device_map()
    assert devices == ["cuda:0", "cuda:1"]

@patch("torch.cuda.is_available")
def test_get_device_map_cpu(mock_is_avail):
    mock_is_avail.return_value = False
    devices = ParallelExecutor.get_device_map()
    assert devices == ["cpu"]

def dummy_task(x):
    return x * 2

def test_run_cpu_parallel():
    data = [1, 2, 3]
    results = ParallelExecutor.run_cpu_parallel(dummy_task, data, use_threads=True, max_workers=2)
    assert sorted(results) == [2, 4, 6]


@patch("thesis_pipeline.utils.parallel_executor.ParallelExecutor.get_device_map", return_value=["cuda:0", "cuda:1"])
@patch("thesis_pipeline.utils.parallel_executor.multiprocessing.get_context")
def test_run_gpu_parallel_raises_on_worker_failure(mock_get_context, _mock_devices):
    class _Process:
        def __init__(self, exitcode):
            self.exitcode = exitcode
            self.pid = 1234 + exitcode

        def start(self):
            return None

        def join(self):
            return None

    ctx = MagicMock()
    ctx.Process.side_effect = [_Process(0), _Process(1)]
    mock_get_context.return_value = ctx

    with pytest.raises(RuntimeError, match="exited with code 1"):
        ParallelExecutor.run_gpu_parallel(lambda *_args, **_kwargs: None, [1, 2, 3, 4])
