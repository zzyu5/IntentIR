from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = ROOT / "scripts" / "flaggems" / "run_gpu_perf_graph.py"


def _load_perf_runner_module():
    spec = importlib.util.spec_from_file_location("run_gpu_perf_graph", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_policy_mode_run_excludes_respects_bench_mode_and_sm() -> None:
    mod = _load_perf_runner_module()

    payload = {
        "run_exclude_kernels_by_bench_mode": {
            "graph": ["graph_only"],
            "eager": ["eager_only"],
        },
        "run_exclude_kernels_by_sm": {
            "sm_90": ["sm90_all_modes"],
        },
        "run_exclude_kernels_by_bench_mode_and_sm": {
            "graph": {
                "sm_90": ["graph_sm90_only"],
            }
        },
    }

    graph_sm90 = mod._policy_mode_run_excludes(payload, bench_mode="graph", cuda_sm="sm90")
    assert graph_sm90 == {"graph_only", "sm90_all_modes", "graph_sm90_only"}

    eager_sm90 = mod._policy_mode_run_excludes(payload, bench_mode="eager", cuda_sm="sm_90")
    assert eager_sm90 == {"eager_only", "sm90_all_modes"}


def test_policy_mode_run_excludes_graph_or_eager_inherits_graph_entries() -> None:
    mod = _load_perf_runner_module()

    payload = {
        "run_exclude_kernels_by_bench_mode": {
            "graph": ["graph_only"],
            "graph_or_eager": ["mixed_only"],
        }
    }
    excludes = mod._policy_mode_run_excludes(payload, bench_mode="graph_or_eager", cuda_sm="sm_90")
    assert excludes == {"graph_only", "mixed_only"}
