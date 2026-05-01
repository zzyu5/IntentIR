from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_prepare_kernel_context_ignores_stale_baseline_npz_and_reruns_flaggems(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import cuda_backend_smoke as mod
    from pipeline.triton.providers.flaggems import specs as flaggems_specs

    report_path = tmp_path / "row_sum.json"
    np.savez(
        tmp_path / "row_sum.baseline.npz",
        inp=np.zeros((4, 64), dtype=np.float32),
        out=np.zeros((4,), dtype=np.float32),
    )
    report_path.write_text(
        json.dumps(
            {
                "kernel": "row_sum",
                "backend_target": "cuda_5090d",
                "flaggems_opset": "deterministic_forward",
                "baseline": {
                    "shapes": {"M": 8, "N": 16},
                    "seed": 0,
                    "skipped": "baseline too large to cache (over 16MB)",
                },
            }
        ),
        encoding="utf-8",
    )

    io_spec = {
        "tensors": {
            "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            "out": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
        },
        "outputs": ["out"],
    }
    intent_json = {
        "name": "row_sum",
        "tensors": dict(io_spec["tensors"]),
        "outputs": ["out"],
        "ops": [{"op": "reduce_sum", "inputs": ["inp"], "output": "out", "attrs": {"dims": [1]}}],
    }

    monkeypatch.setattr(
        mod,
        "_load_intent_and_contract",
        lambda *args, **kwargs: (dict(intent_json), dict(io_spec), {"schema_version": "intent_mlir_backend_contract_v2"}),
    )

    class _FakeSpec:
        def __init__(self) -> None:
            self.name = "row_sum"

        @staticmethod
        def runner(case):
            m = int(case.shapes["M"])
            n = int(case.shapes["N"])
            inp = np.arange(m * n, dtype=np.float32).reshape(m, n)
            out = inp.sum(axis=1, dtype=np.float32)
            return {"inp": inp, "out": out}

    monkeypatch.setattr(flaggems_specs, "default_flaggems_kernel_specs", lambda **kwargs: [_FakeSpec()])
    monkeypatch.setattr(flaggems_specs, "coverage_flaggems_kernel_specs", lambda **kwargs: [])

    ctx = mod._prepare_kernel_context(
        "row_sum",
        frontend="triton",
        triton_provider="flaggems",
        artifact_dir=str(tmp_path),
        require_baseline_npz=False,
    )

    assert ctx["baseline_source"] == "rerun_triton_flaggems"
    assert tuple(np.asarray(ctx["baseline"]["inp"]).shape) == (8, 16)
    assert tuple(np.asarray(ctx["baseline"]["out"]).shape) == (8,)
    assert ctx["bindings"]["M"] == 8
    assert ctx["bindings"]["N"] == 16
