from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = ROOT / "scripts" / "flaggems" / "run_gpu_perf_graph.py"


def _load_perf_runner_module():
    spec = importlib.util.spec_from_file_location("run_gpu_perf_graph", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bench_kernel_in_subprocess_marks_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _load_perf_runner_module()

    class _FakeProc:
        def __init__(self) -> None:
            self.pid = 12345
            self.returncode = None

        def communicate(self, timeout=None):  # noqa: ARG002
            raise subprocess.TimeoutExpired(cmd=["python"], timeout=1.0, output="partial stdout\n")

        def kill(self) -> None:
            self.returncode = -9

    monkeypatch.setattr(mod.subprocess, "Popen", lambda *args, **kwargs: _FakeProc())
    monkeypatch.setattr(mod.os, "killpg", lambda pid, sig: None)
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)

    cli_args = type(
        "_Args",
        (),
        {
            "bench_mode": "graph",
            "warmup": 1,
            "iters": 10,
            "repeats": 2,
            "threshold": 0.9,
            "p50_threshold": 0.0,
            "cuda_runtime_backend": "nvrtc",
            "intent_artifact_dir": str(tmp_path / "intent"),
            "coverage_batches": None,
            "policy_json": None,
            "tuning_db": None,
        },
    )()

    row = mod._bench_kernel_in_subprocess(
        kernel="abs2d",
        family="elementwise_broadcast",
        chunk_name="chunk_001",
        cli_args=cli_args,
        kernel_source="coverage_batches",
        out_root=tmp_path / "out",
        worker_timeout_sec=30.0,
        shape_overrides=None,
    )

    assert row["reason_code"] == "worker_timeout"
    assert row["skip_reason"] == "worker_timeout"
    assert row["count_in_denominator"] is False
    assert row["isolated_worker_timeout_sec"] == 30.0
