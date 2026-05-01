from __future__ import annotations

import json
from pathlib import Path

from intent_ir.mlir.module import IntentMLIRModule
from intent_ir.mlir.pass_manager import _run_one_pass
from pipeline.mlir_backends import select_mlir_backend_route


def test_cpp_plugin_mlir_fallback_runs_python_cuda_passes(monkeypatch) -> None:
    module = IntentMLIRModule(module_text="module {}", dialect_version="std_mlir_v1", meta={})

    def _apply_tuning_db(mod: IntentMLIRModule, **_: object) -> IntentMLIRModule:
        mod = IntentMLIRModule(
            module_text=mod.module_text,
            dialect_version=mod.dialect_version,
            provenance=dict(mod.provenance or {}),
            symbols=list(mod.symbols or []),
            meta=dict(mod.meta or {}),
            intent_json=(dict(mod.intent_json) if isinstance(mod.intent_json, dict) else None),
        )
        mod.meta["apply_tuning_db_seen"] = True
        return mod

    def _lower_cuda(mod: IntentMLIRModule, **_: object) -> IntentMLIRModule:
        mod = IntentMLIRModule(
            module_text=mod.module_text,
            dialect_version=mod.dialect_version,
            provenance=dict(mod.provenance or {}),
            symbols=list(mod.symbols or []),
            meta=dict(mod.meta or {}),
            intent_json=(dict(mod.intent_json) if isinstance(mod.intent_json, dict) else None),
        )
        mod.meta["lower_cuda_seen"] = True
        return mod

    monkeypatch.delenv("INTENTIR_MLIR_PASS_PLUGIN", raising=False)
    monkeypatch.setenv("INTENTIR_AUTO_MLIR_PASS_PLUGIN", "0")
    monkeypatch.setitem(__import__("intent_ir.mlir.pass_manager", fromlist=["PASS_REGISTRY"]).PASS_REGISTRY, "apply_tuning_db", _apply_tuning_db)
    monkeypatch.setitem(__import__("intent_ir.mlir.pass_manager", fromlist=["PASS_REGISTRY"]).PASS_REGISTRY, "lower_intent_to_cuda_gpu_kernel", _lower_cuda)

    result = _run_one_pass(
        module,
        "mlir-opt:pass-pipeline=builtin.module(intentir-apply-tuning-db-cuda-v1,intentir-lower-cuda-focus-v1)",
        backend="cuda",
        toolchain={"tools": {"mlir-opt": {"path": ""}}},
    )
    assert result.kind == "python"
    assert "python_fallback:intentir_mlir_plugin_unavailable" in result.detail
    assert result.module.meta["apply_tuning_db_seen"] is True
    assert result.module.meta["lower_cuda_seen"] is True
    assert result.module.meta["intentir_mlir_opt_fallback_passes"] == [
        "apply_tuning_db",
        "lower_intent_to_cuda_gpu_kernel",
    ]


def test_cpp_plugin_mlir_fallback_runs_extract_gpu_module(monkeypatch) -> None:
    module = IntentMLIRModule(module_text="module { gpu.module @kernels {} }", dialect_version="std_mlir_v1", meta={})

    def _extract_gpu(mod: IntentMLIRModule, **_: object) -> IntentMLIRModule:
        mod = IntentMLIRModule(
            module_text="module {}",
            dialect_version=mod.dialect_version,
            provenance=dict(mod.provenance or {}),
            symbols=list(mod.symbols or []),
            meta=dict(mod.meta or {}),
            intent_json=(dict(mod.intent_json) if isinstance(mod.intent_json, dict) else None),
        )
        mod.meta["extract_gpu_seen"] = True
        return mod

    monkeypatch.delenv("INTENTIR_MLIR_PASS_PLUGIN", raising=False)
    monkeypatch.setenv("INTENTIR_AUTO_MLIR_PASS_PLUGIN", "0")
    monkeypatch.setitem(__import__("intent_ir.mlir.pass_manager", fromlist=["PASS_REGISTRY"]).PASS_REGISTRY, "extract_gpu_module_llvm", _extract_gpu)

    result = _run_one_pass(
        module,
        "mlir-opt:pass-pipeline=builtin.module(intentir-extract-gpu-module-llvm-v1)",
        backend="cuda",
        toolchain={"tools": {"mlir-opt": {"path": ""}}},
    )
    assert result.kind == "python"
    assert result.module.meta["extract_gpu_seen"] is True
    assert result.module.meta["intentir_mlir_opt_fallback_passes"] == ["extract_gpu_module_llvm"]


def _write_cpp_router_state(tmp_path: Path) -> None:
    state_dir = tmp_path / "workflow" / "flaggems" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "compiler_cpp_wave4_shape_gated_kernels.json").write_text(
        json.dumps(
            {
                "schema_version": "intentir_compiler_cpp_shape_gated_wave_v1",
                "wave": "wave4",
                "kernels": ["layer_norm_persistent"],
            }
        ),
        encoding="utf-8",
    )
    (state_dir / "compiler_cpp_shape_policies.json").write_text(
        json.dumps(
            {
                "schema_version": "intentir_compiler_cpp_shape_policy_v1",
                "kernels": {
                    "layer_norm_persistent": {
                        "match": {"N": {"max": 4096}},
                        "reason": "test policy",
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_cpp_router_uses_cpp_path_when_shape_policy_matches(monkeypatch, tmp_path: Path) -> None:
    _write_cpp_router_state(tmp_path)
    monkeypatch.setenv("INTENTIR_COMPILER_STACK", "cpp_plugin")
    monkeypatch.setenv("INTENTIR_COMPILER_CPP_WAVE", "wave4")
    monkeypatch.setenv("INTENTIR_COMPILER_CPP_MISS_POLICY", "strict")
    monkeypatch.delenv("INTENTIR_REAL_MLIR", raising=False)

    route = select_mlir_backend_route(
        "cuda_5090d",
        spec_name="layer_norm_persistent",
        root=tmp_path,
        shape_bindings={"M": 128, "N": 4096},
    )
    assert route.stack_family == "cpp_plugin"
    assert route.llvm_pipeline == "downstream_cuda_std_cpp_llvm"
    assert route.route_reason == "cpp_plugin_shape_wave_hit"
    assert route.route_detail == ""


def test_cpp_router_blocks_cpp_path_when_shape_policy_rejects(monkeypatch, tmp_path: Path) -> None:
    _write_cpp_router_state(tmp_path)
    monkeypatch.setenv("INTENTIR_COMPILER_STACK", "cpp_plugin")
    monkeypatch.setenv("INTENTIR_COMPILER_CPP_WAVE", "wave4")
    monkeypatch.setenv("INTENTIR_COMPILER_CPP_MISS_POLICY", "strict")
    monkeypatch.delenv("INTENTIR_REAL_MLIR", raising=False)

    route = select_mlir_backend_route(
        "cuda_5090d",
        spec_name="layer_norm_persistent",
        root=tmp_path,
        shape_bindings={"M": 64, "N": 8192},
    )
    assert route.stack_family == "cpp_plugin"
    assert route.llvm_pipeline is None
    assert route.route_reason == "compiler_cpp_shape_excludes_kernel"
    assert "N=8192>max=4096" in route.route_detail


def test_cpp_router_falls_back_to_python_real_mlir_when_shape_policy_rejects(monkeypatch, tmp_path: Path) -> None:
    _write_cpp_router_state(tmp_path)
    state_dir = tmp_path / "workflow" / "flaggems" / "state"
    (state_dir / "cuda_real_mlir_wave25_kernels.json").write_text(
        json.dumps(
            {
                "schema_version": "intentir_cuda_real_mlir_wave_v1",
                "wave": "wave25",
                "kernels": ["layer_norm_persistent"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("INTENTIR_COMPILER_STACK", "cpp_plugin")
    monkeypatch.setenv("INTENTIR_COMPILER_CPP_WAVE", "wave4")
    monkeypatch.setenv("INTENTIR_COMPILER_CPP_MISS_POLICY", "strict")
    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")
    monkeypatch.setenv("INTENTIR_CUDA_REAL_MLIR_WAVE", "wave25")

    route = select_mlir_backend_route(
        "cuda_5090d",
        spec_name="layer_norm_persistent",
        root=tmp_path,
        shape_bindings={"M": 64, "N": 8192},
    )
    assert route.stack_family == "python_mlir"
    assert route.llvm_pipeline == "downstream_cuda_std_llvm"
    assert route.route_reason == "cpp_plugin_shape_fallback_python_real_mlir"
    assert route.used_fallback is True
    assert "N=8192>max=4096" in route.route_detail


def test_cpp_router_blocks_shape_policy_only_kernel_when_not_wave_admitted(monkeypatch, tmp_path: Path) -> None:
    state_dir = tmp_path / "workflow" / "flaggems" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "compiler_cpp_shape_policies.json").write_text(
        json.dumps(
            {
                "schema_version": "intentir_compiler_cpp_shape_policy_v1",
                "kernels": {
                    "layer_norm_persistent": {
                        "match": {"N": {"max": 4096}},
                        "reason": "test policy",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("INTENTIR_COMPILER_STACK", "cpp_plugin")
    monkeypatch.setenv("INTENTIR_COMPILER_CPP_WAVE", "wave4")
    monkeypatch.setenv("INTENTIR_COMPILER_CPP_MISS_POLICY", "strict")

    route = select_mlir_backend_route(
        "cuda_5090d",
        spec_name="layer_norm_persistent",
        root=tmp_path,
        shape_bindings={"M": 128, "N": 4096},
    )
    assert route.stack_family == "cpp_plugin"
    assert route.llvm_pipeline is None
    assert route.route_reason == "compiler_cpp_shape_policy_only_kernel"
    assert route.route_detail == "shape_policy_present_without_wave_admission"
