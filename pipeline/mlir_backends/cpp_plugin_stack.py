from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


_CPP_WAVE_KERNELS: dict[tuple[str, str], set[str]] = {}
_CPP_SHAPE_GATED_WAVE_KERNELS: dict[tuple[str, str], set[str]] = {}
_CPP_SHAPE_POLICIES: dict[str, dict[str, dict[str, Any]]] = {}


def compiler_cpp_wave_name() -> str:
    return str(os.getenv("INTENTIR_COMPILER_CPP_WAVE", "wave2")).strip().lower()


def compiler_cpp_miss_policy() -> str:
    return str(os.getenv("INTENTIR_COMPILER_CPP_MISS_POLICY", "skip")).strip().lower()


def _load_kernel_set(path: Path) -> set[str]:
    kernels: set[str] = set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = payload.get("kernels")
            if isinstance(rows, list):
                for item in rows:
                    name = str(item).strip()
                    if name:
                        kernels.add(name)
    except Exception:
        kernels = set()
    return kernels


def compiler_cpp_wave_kernels(*, root: Path, wave: str | None = None) -> set[str]:
    wave_name = str(wave or compiler_cpp_wave_name()).strip().lower()
    if not wave_name:
        return set()
    key = (str(root), wave_name)
    cached = _CPP_WAVE_KERNELS.get(key)
    if cached is not None:
        return cached
    path = Path(root) / "workflow" / "flaggems" / "state" / f"compiler_cpp_{wave_name}_kernels.json"
    kernels = _load_kernel_set(path)
    _CPP_WAVE_KERNELS[key] = kernels
    return kernels


def compiler_cpp_shape_gated_wave_kernels(*, root: Path, wave: str | None = None) -> set[str]:
    wave_name = str(wave or compiler_cpp_wave_name()).strip().lower()
    if not wave_name:
        return set()
    key = (str(root), wave_name)
    cached = _CPP_SHAPE_GATED_WAVE_KERNELS.get(key)
    if cached is not None:
        return cached
    path = Path(root) / "workflow" / "flaggems" / "state" / f"compiler_cpp_{wave_name}_shape_gated_kernels.json"
    kernels = _load_kernel_set(path)
    _CPP_SHAPE_GATED_WAVE_KERNELS[key] = kernels
    return kernels


def compiler_cpp_wave_admission(*, root: Path, spec_name: str, wave: str | None = None) -> str:
    spec = str(spec_name or "").strip()
    wave_name = str(wave or compiler_cpp_wave_name()).strip().lower()
    if not spec or not wave_name:
        return ""
    if spec in compiler_cpp_shape_gated_wave_kernels(root=root, wave=wave_name):
        return "shape_gated"
    if spec in compiler_cpp_wave_kernels(root=root, wave=wave_name):
        return "admitted"
    return ""


def compiler_cpp_shape_policies(*, root: Path) -> dict[str, dict[str, Any]]:
    root_key = str(Path(root))
    cached = _CPP_SHAPE_POLICIES.get(root_key)
    if cached is not None:
        return cached

    path = Path(root) / "workflow" / "flaggems" / "state" / "compiler_cpp_shape_policies.json"
    policies: dict[str, dict[str, Any]] = {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            kernels = payload.get("kernels")
            if isinstance(kernels, dict):
                for raw_name, raw_policy in kernels.items():
                    name = str(raw_name).strip()
                    if not name or not isinstance(raw_policy, dict):
                        continue
                    policies[name] = dict(raw_policy)
    except Exception:
        policies = {}

    _CPP_SHAPE_POLICIES[root_key] = policies
    return policies


def compiler_cpp_shape_supported(
    *,
    root: Path,
    spec_name: str,
    shape_bindings: dict[str, int] | None = None,
) -> tuple[bool, str]:
    spec = str(spec_name or "").strip()
    if not spec:
        return True, ""

    policy = dict(compiler_cpp_shape_policies(root=root).get(spec) or {})
    match = policy.get("match")
    if not isinstance(match, dict) or not match:
        return True, ""

    bindings: dict[str, int] = {}
    for raw_key, raw_value in dict(shape_bindings or {}).items():
        key = str(raw_key).strip()
        if not key:
            continue
        try:
            bindings[key] = int(raw_value)
        except Exception:
            continue
    if not bindings:
        return False, "shape_bindings_missing"

    detail = ""
    for raw_axis, raw_rule in match.items():
        axis = str(raw_axis).strip()
        if not axis or not isinstance(raw_rule, dict):
            continue
        if axis not in bindings:
            detail = f"{axis}=missing"
            break
        value = int(bindings[axis])
        if raw_rule.get("eq") is not None and value != int(raw_rule.get("eq")):
            detail = f"{axis}={value}!=eq={int(raw_rule.get('eq'))}"
            break
        if raw_rule.get("min") is not None and value < int(raw_rule.get("min")):
            detail = f"{axis}={value}<min={int(raw_rule.get('min'))}"
            break
        if raw_rule.get("max") is not None and value > int(raw_rule.get("max")):
            detail = f"{axis}={value}>max={int(raw_rule.get('max'))}"
            break
        allowed = raw_rule.get("in")
        if isinstance(allowed, list):
            allowed_ints = []
            for item in allowed:
                try:
                    allowed_ints.append(int(item))
                except Exception:
                    continue
            if allowed_ints and value not in allowed_ints:
                detail = f"{axis}={value} not_in={allowed_ints}"
                break

    if not detail:
        return True, ""

    reason = str(policy.get("reason") or "").strip()
    return False, (f"{detail}; {reason}" if reason else detail)
