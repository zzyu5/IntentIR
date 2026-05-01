from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cpp_plugin_stack import (
    compiler_cpp_miss_policy,
    compiler_cpp_shape_policies,
    compiler_cpp_shape_supported,
    compiler_cpp_wave_admission,
    compiler_cpp_wave_kernels,
    compiler_cpp_wave_name,
)
from .python_stack import (
    cuda_real_mlir_wave_kernels,
    cuda_real_mlir_wave_name,
    rvv_real_mlir_wave_kernels,
    rvv_real_mlir_wave_name,
)


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class MlirBackendRoute:
    stack_name: str
    stack_family: str
    backend_target: str
    spec_name: str
    llvm_pipeline: str | None
    llvm_backend: str | None
    wave_name: str
    miss_policy: str
    route_reason: str
    used_fallback: bool = False
    route_detail: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "stack_name": str(self.stack_name),
            "stack_family": str(self.stack_family),
            "backend_target": str(self.backend_target),
            "spec_name": str(self.spec_name),
            "llvm_pipeline": (str(self.llvm_pipeline) if self.llvm_pipeline is not None else ""),
            "llvm_backend": (str(self.llvm_backend) if self.llvm_backend is not None else ""),
            "wave_name": str(self.wave_name),
            "miss_policy": str(self.miss_policy),
            "route_reason": str(self.route_reason),
            "used_fallback": bool(self.used_fallback),
            "route_detail": str(self.route_detail),
        }


def compiler_stack_name() -> str:
    return str(os.getenv("INTENTIR_COMPILER_STACK", "python")).strip().lower()


def _real_mlir_enabled() -> bool:
    return str(os.getenv("INTENTIR_REAL_MLIR", "")).strip().lower() in {"1", "true", "yes", "on"}


def _cuda_real_mlir_allow_unknown() -> bool:
    raw = str(os.getenv("INTENTIR_CUDA_REAL_MLIR_ALLOW_UNKNOWN", "0") or "").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def select_mlir_backend_route(
    backend_target: str | None,
    *,
    spec_name: str | None = None,
    root: Path | None = None,
    shape_bindings: dict[str, int] | None = None,
) -> MlirBackendRoute:
    repo_root = Path(root or ROOT)
    target = str(backend_target or "").strip().lower()
    spec = str(spec_name or "").strip()
    stack = compiler_stack_name()
    miss_policy = compiler_cpp_miss_policy()
    real_mlir_enabled = _real_mlir_enabled()
    normalized_shape_bindings: dict[str, int] = {}
    for raw_key, raw_value in dict(shape_bindings or {}).items():
        key = str(raw_key).strip()
        if not key:
            continue
        try:
            normalized_shape_bindings[key] = int(raw_value)
        except Exception:
            continue

    if not target:
        return MlirBackendRoute(
            stack_name=stack,
            stack_family=("cpp_plugin" if stack in {"cpp", "cpp_plugin", "c++"} else "python_mlir"),
            backend_target="",
            spec_name=spec,
            llvm_pipeline=None,
            llvm_backend=None,
            wave_name="",
            miss_policy=miss_policy,
            route_reason="backend_target_not_set",
        )

    if stack in {"cpp", "cpp_plugin", "c++"}:
        cpp_wave = compiler_cpp_wave_name()
        cpp_admission = compiler_cpp_wave_admission(root=repo_root, wave=cpp_wave, spec_name=spec) if cpp_wave else ""
        shape_policy_present = bool(spec and compiler_cpp_shape_policies(root=repo_root).get(spec))
        if spec and cpp_admission:
            if cpp_admission == "shape_gated" and not shape_policy_present:
                return MlirBackendRoute(
                    stack_name=stack,
                    stack_family="cpp_plugin",
                    backend_target=target,
                    spec_name=spec,
                    llvm_pipeline=None,
                    llvm_backend=None,
                    wave_name=cpp_wave,
                    miss_policy=miss_policy,
                    route_reason="compiler_cpp_shape_policy_missing",
                    route_detail="shape_gated_wave_admission_requires_shape_policy",
                )
            shape_ok, shape_detail = compiler_cpp_shape_supported(
                root=repo_root,
                spec_name=spec,
                shape_bindings=normalized_shape_bindings,
            )
            if not shape_ok:
                if target.startswith("cuda") and real_mlir_enabled:
                    wave = cuda_real_mlir_wave_name(real_mlir_enabled=real_mlir_enabled)
                    kernels = cuda_real_mlir_wave_kernels(root=repo_root, wave=wave) if wave else set()
                    if spec and spec in kernels:
                        return MlirBackendRoute(
                            stack_name=stack,
                            stack_family="python_mlir",
                            backend_target=target,
                            spec_name=spec,
                            llvm_pipeline="downstream_cuda_std_llvm",
                            llvm_backend="cuda",
                            wave_name=wave,
                            miss_policy=miss_policy,
                            route_reason="cpp_plugin_shape_fallback_python_real_mlir",
                            used_fallback=True,
                            route_detail=shape_detail,
                        )
                if target.startswith("rvv") and real_mlir_enabled:
                    wave = rvv_real_mlir_wave_name(real_mlir_enabled=real_mlir_enabled)
                    kernels = rvv_real_mlir_wave_kernels(root=repo_root, wave=wave) if wave else set()
                    if spec and spec in kernels:
                        return MlirBackendRoute(
                            stack_name=stack,
                            stack_family="python_mlir",
                            backend_target=target,
                            spec_name=spec,
                            llvm_pipeline="downstream_rvv_std_llvm",
                            llvm_backend="rvv",
                            wave_name=wave,
                            miss_policy=miss_policy,
                            route_reason="cpp_plugin_shape_fallback_python_real_mlir",
                            used_fallback=True,
                            route_detail=shape_detail,
                        )
                return MlirBackendRoute(
                    stack_name=stack,
                    stack_family="cpp_plugin",
                    backend_target=target,
                    spec_name=spec,
                    llvm_pipeline=None,
                    llvm_backend=None,
                    wave_name=cpp_wave,
                    miss_policy=miss_policy,
                    route_reason="compiler_cpp_shape_excludes_kernel",
                    route_detail=shape_detail,
                )
            if target.startswith("rvv"):
                return MlirBackendRoute(
                    stack_name=stack,
                    stack_family="cpp_plugin",
                    backend_target=target,
                    spec_name=spec,
                    llvm_pipeline="downstream_rvv_std_llvm_cpp",
                    llvm_backend="rvv",
                    wave_name=cpp_wave,
                    miss_policy=miss_policy,
                    route_reason=("cpp_plugin_shape_wave_hit" if cpp_admission == "shape_gated" else "cpp_plugin_wave_hit"),
                )
            if target.startswith("cuda"):
                return MlirBackendRoute(
                    stack_name=stack,
                    stack_family="cpp_plugin",
                    backend_target=target,
                    spec_name=spec,
                    llvm_pipeline="downstream_cuda_std_cpp_llvm",
                    llvm_backend="cuda",
                    wave_name=cpp_wave,
                    miss_policy=miss_policy,
                    route_reason=("cpp_plugin_shape_wave_hit" if cpp_admission == "shape_gated" else "cpp_plugin_wave_hit"),
                )
        if miss_policy not in {"python", "py"}:
            return MlirBackendRoute(
                stack_name=stack,
                stack_family="cpp_plugin",
                backend_target=target,
                spec_name=spec,
                llvm_pipeline=None,
                llvm_backend=None,
                wave_name=cpp_wave,
                miss_policy=miss_policy,
                route_reason=("compiler_cpp_shape_policy_only_kernel" if shape_policy_present else "compiler_cpp_wave_excludes_kernel"),
                route_detail=("shape_policy_present_without_wave_admission" if shape_policy_present else ""),
            )

    if target.startswith("cuda"):
        wave = cuda_real_mlir_wave_name(real_mlir_enabled=real_mlir_enabled)
        if wave and real_mlir_enabled:
            kernels = cuda_real_mlir_wave_kernels(root=repo_root, wave=wave)
            if spec and spec in kernels:
                return MlirBackendRoute(
                    stack_name=stack,
                    stack_family="python_mlir",
                    backend_target=target,
                    spec_name=spec,
                    llvm_pipeline="downstream_cuda_std_llvm",
                    llvm_backend="cuda",
                    wave_name=wave,
                    miss_policy=miss_policy,
                    route_reason="python_real_mlir_wave_hit",
                    used_fallback=bool(stack in {"cpp", "cpp_plugin", "c++"}),
                )
            if _cuda_real_mlir_allow_unknown():
                return MlirBackendRoute(
                    stack_name=stack,
                    stack_family="python_mlir",
                    backend_target=target,
                    spec_name=spec,
                    llvm_pipeline="downstream_cuda_std_llvm",
                    llvm_backend="cuda",
                    wave_name=wave,
                    miss_policy=miss_policy,
                    route_reason="python_real_mlir_allow_unknown",
                    used_fallback=bool(stack in {"cpp", "cpp_plugin", "c++"}),
                )
            return MlirBackendRoute(
                stack_name=stack,
                stack_family="python_mlir",
                backend_target=target,
                spec_name=spec,
                llvm_pipeline=None,
                llvm_backend=None,
                wave_name=wave,
                miss_policy=miss_policy,
                route_reason="cuda_real_mlir_wave_excludes_kernel",
                used_fallback=bool(stack in {"cpp", "cpp_plugin", "c++"}),
            )
        return MlirBackendRoute(
            stack_name=stack,
            stack_family="python_mlir",
            backend_target=target,
            spec_name=spec,
            llvm_pipeline="downstream_cuda_llvm",
            llvm_backend="cuda",
            wave_name=wave,
            miss_policy=miss_policy,
            route_reason="legacy_cuda_llvm",
            used_fallback=bool(stack in {"cpp", "cpp_plugin", "c++"}),
        )

    if target.startswith("rvv"):
        wave = rvv_real_mlir_wave_name(real_mlir_enabled=real_mlir_enabled)
        if wave and real_mlir_enabled:
            kernels = rvv_real_mlir_wave_kernels(root=repo_root, wave=wave)
            if spec and spec in kernels:
                return MlirBackendRoute(
                    stack_name=stack,
                    stack_family="python_mlir",
                    backend_target=target,
                    spec_name=spec,
                    llvm_pipeline="downstream_rvv_std_llvm",
                    llvm_backend="rvv",
                    wave_name=wave,
                    miss_policy=miss_policy,
                    route_reason="python_real_mlir_wave_hit",
                    used_fallback=bool(stack in {"cpp", "cpp_plugin", "c++"}),
                )
            return MlirBackendRoute(
                stack_name=stack,
                stack_family="python_mlir",
                backend_target=target,
                spec_name=spec,
                llvm_pipeline=None,
                llvm_backend=None,
                wave_name=wave,
                miss_policy=miss_policy,
                route_reason="rvv_real_mlir_wave_excludes_kernel",
                used_fallback=bool(stack in {"cpp", "cpp_plugin", "c++"}),
            )
        return MlirBackendRoute(
            stack_name=stack,
            stack_family="python_mlir",
            backend_target=target,
            spec_name=spec,
            llvm_pipeline="downstream_rvv_llvm",
            llvm_backend="rvv",
            wave_name=wave,
            miss_policy=miss_policy,
            route_reason="legacy_rvv_llvm",
            used_fallback=bool(stack in {"cpp", "cpp_plugin", "c++"}),
        )

    return MlirBackendRoute(
        stack_name=stack,
        stack_family=("cpp_plugin" if stack in {"cpp", "cpp_plugin", "c++"} else "python_mlir"),
        backend_target=target,
        spec_name=spec,
        llvm_pipeline=None,
        llvm_backend=None,
        wave_name="",
        miss_policy=miss_policy,
        route_reason="unsupported_backend_target",
        used_fallback=False,
    )


def emit_route_log(prefix: str, route: MlirBackendRoute) -> None:
    pipeline = str(route.llvm_pipeline or "disabled")
    backend = str(route.llvm_backend or "-")
    detail = f" detail={route.route_detail}" if str(route.route_detail or "").strip() else ""
    print(
        f"[{prefix}][mlir-router] stack={route.stack_name} path={route.stack_family} "
        f"pipeline={pipeline} backend={backend} wave={route.wave_name or '-'} "
        f"reason={route.route_reason}{detail} kernel={route.spec_name or '-'}"
    )
