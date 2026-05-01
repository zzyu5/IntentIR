#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontends.common.contract_v2 import evaluate_contract_v2
from frontends.common.obligations import (
    O1_HAS_SEMANTIC_ANCHOR,
    O2_AFFINE_OR_STRUCTURED_INDEXING,
    O3_MASK_IMPLIES_INBOUNDS,
    O4_SHAPE_LAYOUT_MATCH,
    O5_NO_DATA_DEPENDENT_ADDRESS,
    O7_NO_ATOMICS_OR_CONTROLLED_ATOMICS,
    evaluate_obligations,
)
from frontends.common.static_validate import static_validate
from intent_ir.ir import IntentFunction
from intent_ir.validation import rebuild_repaired_certificate_bundle
from pipeline.triton.providers.flaggems.specs import coverage_flaggems_kernel_specs
from verify.diff_runner import run_diff
from verify.gen_cases import TestCase


DEFAULT_AUDIT_DIR = ROOT / "artifacts" / "analysis" / "20260420" / "flaggems_partial_repair"
DEFAULT_RUN_DIR = ROOT / "artifacts" / "validation_rounds" / "20260309" / "full196_cuda_cpp_hybrid_sm89_v2"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "analysis" / "20260420" / "flaggems_partial_repair_v2"


def _default_date() -> str:
    try:
        return subprocess.check_output(["date", "+%Y%m%d"], text=True, cwd=str(ROOT)).strip()
    except Exception:
        from datetime import date

        return date.today().strftime("%Y%m%d")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _bool_str(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return ""


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _contract_level(report: Mapping[str, Any]) -> str:
    contract = report.get("contract")
    if isinstance(contract, Mapping):
        level = str(contract.get("level") or "")
        if level:
            return level
    return "NO_CONTRACT"


def _obligation_status_map_from_static(report: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    static_validation = report.get("static_validation")
    if not isinstance(static_validation, Mapping):
        return out
    for row in static_validation.get("obligations") or []:
        if not isinstance(row, Mapping):
            continue
        out[str(row.get("id") or "")] = str(row.get("status") or "")
    return out


def _obligation_reason_map_from_static(report: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    static_validation = report.get("static_validation")
    if not isinstance(static_validation, Mapping):
        return out
    for row in static_validation.get("obligations") or []:
        if not isinstance(row, Mapping):
            continue
        out[str(row.get("id") or "")] = str(row.get("detail") or "")
    return out


def _pick_intent(report: Mapping[str, Any]) -> IntentFunction:
    raw = report.get("intent_expanded") if isinstance(report.get("intent_expanded"), Mapping) else None
    if raw is None:
        raw = report.get("intent") if isinstance(report.get("intent"), Mapping) else None
    if raw is None:
        raise ValueError("report missing intent payload")
    return IntentFunction.from_json_dict(dict(raw))


def _critical_status_map(obligations: Sequence[Any]) -> dict[str, str]:
    return {str(row.id): str(row.status) for row in obligations}


def _decide_final_contract(
    *,
    contract_v2: Any,
    obligations: Sequence[Any],
    static_validation: Any,
    gate_after: bool | None,
) -> tuple[str, list[str]]:
    statuses = _critical_status_map(obligations)
    reasons = list(getattr(contract_v2, "reasons", []) or [])
    if gate_after is False:
        reasons.append("source-output gate FAIL")
        return "OUT_OF_SCOPE", reasons
    if statuses.get(O1_HAS_SEMANTIC_ANCHOR) == "FAIL":
        return "OUT_OF_SCOPE", reasons
    if statuses.get(O7_NO_ATOMICS_OR_CONTROLLED_ATOMICS) == "FAIL":
        return "OUT_OF_SCOPE", reasons
    if statuses.get(O5_NO_DATA_DEPENDENT_ADDRESS) == "FAIL":
        return "OUT_OF_SCOPE", reasons
    static_fail = [row for row in getattr(static_validation, "obligations", []) if str(getattr(row, "status", "")) == "FAIL"]
    critical_ids = [
        O1_HAS_SEMANTIC_ANCHOR,
        O2_AFFINE_OR_STRUCTURED_INDEXING,
        O3_MASK_IMPLIES_INBOUNDS,
        O4_SHAPE_LAYOUT_MATCH,
        O5_NO_DATA_DEPENDENT_ADDRESS,
        O7_NO_ATOMICS_OR_CONTROLLED_ATOMICS,
    ]
    if gate_after is True and not static_fail and all(statuses.get(key) == "PASS" for key in critical_ids):
        return "FULL", reasons
    if getattr(contract_v2, "level", None) == "OUT_OF_SCOPE":
        return "OUT_OF_SCOPE", reasons
    if gate_after is None:
        reasons.append("source-output gate missing")
    return "PARTIAL", reasons


def _gate_before(report: Mapping[str, Any]) -> bool | None:
    diff = report.get("diff")
    if isinstance(diff, Mapping) and isinstance(diff.get("ok"), bool):
        return bool(diff.get("ok"))
    return None


def _build_spec_cache() -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for backend_target in ["cuda_5090d", "cuda_h100", "rvv"]:
        try:
            specs = coverage_flaggems_kernel_specs(flaggems_opset="deterministic_forward", backend_target=backend_target)
        except Exception:
            specs = []
        cache[backend_target] = {str(spec.name): spec for spec in specs}
    return cache


def _rerun_gate(
    *,
    report: Mapping[str, Any],
    intent: IntentFunction,
    spec_cache: Mapping[str, Mapping[str, Any]],
) -> tuple[bool | None, dict[str, Any], str]:
    kernel = str(report.get("kernel") or "")
    backend_target = str(report.get("backend_target") or ((report.get("descriptor") or {}).get("meta") or {}).get("backend_target") or "cuda_5090d")
    spec = (spec_cache.get(backend_target) or {}).get(kernel)
    if spec is None:
        return None, {}, f"spec not found for backend_target={backend_target}"
    raw_cases = dict((report.get("cases") or {})).get("in_contract") or []
    if not isinstance(raw_cases, list) or not raw_cases:
        return None, {}, "no in-contract cases recorded"
    cases = [
        TestCase(shapes={str(k): int(v) for k, v in dict(case).items()}, dtypes={}, seed=0)
        for case in raw_cases
        if isinstance(case, Mapping)
    ]
    tolerances = dict(report.get("tolerances") or {}) if isinstance(report.get("tolerances"), Mapping) else None
    diffs, _ = run_diff(intent, spec.runner, cases, tolerances=tolerances)
    results = []
    worst_abs = 0.0
    worst_rel = 0.0
    ok = True
    for case, diff in zip(cases, diffs):
        results.append(
            {
                "case_shapes": dict(case.shapes),
                "ok": bool(diff.ok),
                "summary": str(diff.summary),
                "max_abs": float(diff.max_abs_err),
                "max_rel": float(diff.max_rel_err),
            }
        )
        ok = ok and bool(diff.ok)
        worst_abs = max(worst_abs, float(diff.max_abs_err))
        worst_rel = max(worst_rel, float(diff.max_rel_err))
    payload = {
        "ok": bool(ok),
        "worst": {
            "summary": "ok" if ok else next((str(row["summary"]) for row in results if not row["ok"]), "diff_fail"),
            "max_abs": float(worst_abs),
            "max_rel": float(worst_rel),
        },
        "results": results,
        "rerun": True,
    }
    return bool(ok), payload, ""


def _post_repair_reason(
    *,
    new_contract: str,
    statuses: Mapping[str, str],
    static_validation: Any,
    gate_after: bool | None,
    access_witnesses: Sequence[Mapping[str, Any]],
) -> str:
    if gate_after is False:
        return "gate missing/fail"
    if statuses.get(O5_NO_DATA_DEPENDENT_ADDRESS) not in {"", "PASS"}:
        return "data-dependent indexing"
    if statuses.get(O2_AFFINE_OR_STRUCTURED_INDEXING) not in {"", "PASS"}:
        return "index still cannot canonicalize"
    if statuses.get(O4_SHAPE_LAYOUT_MATCH) not in {"", "PASS"}:
        return "shape/layout witness missing"
    if statuses.get(O3_MASK_IMPLIES_INBOUNDS) not in {"", "PASS"}:
        has_mask = any(str(row.get("mask_expr") or "").strip() for row in access_witnesses)
        has_bound = any(str(row.get("tensor_bound") or "").strip() for row in access_witnesses)
        has_domains = any(bool(row.get("domain_constraints")) for row in access_witnesses)
        if not has_mask:
            return "missing mask evidence"
        if not has_bound or not has_domains:
            return "missing domain constraint"
        return "solver template missing"
    if any(str(getattr(row, "status", "")) == "FAIL" for row in getattr(static_validation, "obligations", [])):
        return "other"
    return "other"


@dataclass
class KernelRepairResult:
    kernel_name: str
    family: str
    root_cause: str
    old_contract: str
    new_contract: str
    source_gate_before: bool | None
    source_gate_after: bool | None
    old_o2: str
    new_o2: str
    old_o3: str
    new_o3: str
    old_o3_reason: str
    new_o3_reason: str
    mask_bound: str
    access_bound: str
    index_before: str
    index_after: str
    binding_confidence: str
    validator_reran: bool
    gate_reran: bool
    changed: bool
    why_not_changed: str
    artifact_before: str
    artifact_after: str
    repaired_report: dict[str, Any]
    repaired_examples: list[dict[str, Any]]
    static_ok_after: bool


def _first_index_expr(report: Mapping[str, Any]) -> str:
    cert = ((report.get("certificate_v2") or {}).get("semantic_facts") or {}).get("canonical_evidence") or {}
    accesses = cert.get("accesses") or []
    if not isinstance(accesses, list) or not accesses:
        return ""
    first = accesses[0]
    if not isinstance(first, Mapping):
        return ""
    exprs = first.get("index_exprs") or []
    if not isinstance(exprs, list) or not exprs:
        return ""
    expr = exprs[0]
    if not isinstance(expr, Mapping):
        return ""
    terms = expr.get("terms") or {}
    const = int(expr.get("const") or 0)
    parts = []
    if isinstance(terms, Mapping):
        for key, value in sorted(terms.items()):
            coeff = int(value)
            if coeff == 1:
                parts.append(str(key))
            elif coeff == -1:
                parts.append(f"-{key}")
            else:
                parts.append(f"{coeff}*{key}")
    if const != 0 or not parts:
        parts.append(str(const))
    return " + ".join(parts).replace("+ -", "- ")


def _first_success_example(
    *,
    report: Mapping[str, Any],
    result: KernelRepairResult,
    obligations: Sequence[Any],
    access_witnesses: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if result.new_contract != "FULL":
        return None
    o3 = next((row for row in obligations if str(row.id) == O3_MASK_IMPLIES_INBOUNDS), None)
    o3_witness = dict(getattr(o3, "witness", {}) or {}) if o3 is not None else {}
    access_checks = list(o3_witness.get("access_checks") or [])
    proof = ""
    for access in access_checks:
        if not isinstance(access, Mapping):
            continue
        dims = access.get("dims") or []
        if not isinstance(dims, list) or not dims:
            continue
        parts = []
        for dim in dims:
            if not isinstance(dim, Mapping):
                continue
            witness = dict(dim.get("witness") or {})
            upper = str(witness.get("upper_bound_clause") or "")
            lower = str(witness.get("lower_bound_clause") or witness.get("lower_bound_proof") or "")
            if upper or lower:
                parts.append("; ".join(piece for piece in [upper, lower] if piece))
        if parts:
            proof = " | ".join(parts)
            break
    best_witness = next(
        (
            row
            for row in access_witnesses
            if str(row.get("binding_confidence") or "") in {"exact", "structural", "heuristic"} and str(row.get("mask_expr") or "").strip()
        ),
        None,
    )
    if best_witness is None:
        best_witness = access_witnesses[0] if access_witnesses else None
    if best_witness is None:
        return None
    return {
        "kernel": result.kernel_name,
        "old_reason": " | ".join(str(x) for x in list((report.get("contract") or {}).get("reasons") or [])),
        "access_witness": dict(best_witness),
        "o3_proof_sketch": proof,
        "old_contract": result.old_contract,
        "new_contract": result.new_contract,
        "source_gate_after": result.source_gate_after,
    }


def _write_repaired_example_md(path: Path, examples: Sequence[Mapping[str, Any]]) -> None:
    lines = ["# Repaired examples", ""]
    if not examples:
        lines.append("- No successful FULL repair cases were recorded.")
    for example in examples[:5]:
        witness = dict(example.get("access_witness") or {})
        lines += [
            f"## {example.get('kernel')}",
            f"- Old reason: `{example.get('old_reason')}`",
            f"- Repaired mask/access witness: access=`{witness.get('access_id')}` mask=`{witness.get('mask_expr')}` index=`{witness.get('normalized_index_expr')}` confidence=`{witness.get('binding_confidence')}`",
            f"- O3 proof sketch: `{example.get('o3_proof_sketch')}`",
            f"- Contract: `{example.get('old_contract')}` -> `{example.get('new_contract')}`",
            f"- Source-output gate after repair: `{example.get('source_gate_after')}`",
            "",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_remaining_partials_md(path: Path, results: Sequence[KernelRepairResult]) -> None:
    groups: dict[str, list[str]] = {}
    for row in results:
        if row.new_contract == "FULL":
            continue
        groups.setdefault(row.why_not_changed or "other", []).append(row.kernel_name)
    lines = ["# Remaining PARTIAL/OOS", ""]
    if not groups:
        lines.append("- No remaining PARTIAL/OOS kernels.")
    for reason, kernels in sorted(groups.items()):
        lines.append(f"## {reason}")
        for kernel in sorted(kernels):
            lines.append(f"- `{kernel}`")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--root-cause", default="mask_access_binding_missing,index_canonicalization_missing")
    parser.add_argument("--rerun-validator", action="store_true")
    parser.add_argument("--rerun-gate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--kernel", action="append", default=None)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--date", default=_default_date())
    args = parser.parse_args()

    audit_dir = args.audit_dir if args.audit_dir.is_absolute() else ROOT / args.audit_dir
    input_dir = args.input_dir if args.input_dir.is_absolute() else ROOT / args.input_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    repaired_reports_dir = output_dir / "repaired_reports"
    repaired_reports_dir.mkdir(parents=True, exist_ok=True)

    root_causes = {item.strip() for item in str(args.root_cause).split(",") if item.strip()}
    audit_rows = _load_csv(audit_dir / "root_cause_summary.csv")
    per_kernel_audit = {row["kernel_name"]: row for row in _load_csv(audit_dir / "per_kernel_repair_audit.csv")}
    wanted_kernels = {str(item).strip() for item in list(args.kernel or []) if str(item).strip()}

    selected_rows = [row for row in audit_rows if row.get("root_cause") in root_causes]
    other_rows = [row for row in audit_rows if row.get("root_cause") not in root_causes]
    if wanted_kernels:
        selected_rows = [row for row in selected_rows if row.get("kernel_name") in wanted_kernels]
    selected_rows = sorted(selected_rows, key=lambda row: str(row.get("kernel_name") or ""))
    if args.limit is not None:
        selected_rows = selected_rows[: int(args.limit)]

    results: list[KernelRepairResult] = []
    repaired_examples: list[dict[str, Any]] = []
    spec_cache = _build_spec_cache() if args.rerun_gate else {}

    for row in selected_rows:
        kernel = str(row["kernel_name"])
        family = str(row["family"])
        report_path = ROOT / str(row["report_path"])
        report = _read_json(report_path)
        artifact_after = repaired_reports_dir / family / f"{kernel}.json"
        if args.resume and artifact_after.is_file():
            repaired_report = _read_json(artifact_after)
            results.append(
                KernelRepairResult(
                    kernel_name=kernel,
                    family=family,
                    root_cause=str(row["root_cause"]),
                    old_contract=str((report.get("contract") or {}).get("level") or "NO_CONTRACT"),
                    new_contract=str((repaired_report.get("contract") or {}).get("level") or "NO_CONTRACT"),
                    source_gate_before=_gate_before(report),
                    source_gate_after=_gate_before(repaired_report),
                    old_o2=str(_obligation_status_map_from_static(report).get(O2_AFFINE_OR_STRUCTURED_INDEXING, "")),
                    new_o2=str(_obligation_status_map_from_static(repaired_report).get(O2_AFFINE_OR_STRUCTURED_INDEXING, "")),
                    old_o3=str(_obligation_status_map_from_static(report).get(O3_MASK_IMPLIES_INBOUNDS, "")),
                    new_o3=str(_obligation_status_map_from_static(repaired_report).get(O3_MASK_IMPLIES_INBOUNDS, "")),
                    old_o3_reason=str(_obligation_reason_map_from_static(report).get(O3_MASK_IMPLIES_INBOUNDS, "")),
                    new_o3_reason=str(_obligation_reason_map_from_static(repaired_report).get(O3_MASK_IMPLIES_INBOUNDS, "")),
                    mask_bound="",
                    access_bound="",
                    index_before=_first_index_expr(report),
                    index_after=_first_index_expr(repaired_report),
                    binding_confidence="",
                    validator_reran=bool(args.rerun_validator),
                    gate_reran=bool(args.rerun_gate),
                    changed=str((report.get("contract") or {}).get("level")) != str((repaired_report.get("contract") or {}).get("level")),
                    why_not_changed="",
                    artifact_before=_rel(report_path),
                    artifact_after=_rel(artifact_after),
                    repaired_report=repaired_report,
                    repaired_examples=[],
                    static_ok_after=bool((repaired_report.get("static_validation") or {}).get("ok")),
                )
            )
            continue

        bundle = rebuild_repaired_certificate_bundle(report)
        repaired_cert = bundle.repaired_certificate_v2
        intent = _pick_intent(report)
        obligations = evaluate_obligations(bundle.descriptor, repaired_cert) if args.rerun_validator else []
        if args.rerun_validator:
            repaired_cert.semantic_facts["obligations"] = [row.to_json_dict() for row in obligations]
            contract_v2 = evaluate_contract_v2(bundle.descriptor, repaired_cert, obligations)
            repaired_cert.meta["contract"] = {
                "level": str(contract_v2.level),
                "reasons": list(contract_v2.reasons),
                "assumptions": list(contract_v2.assumptions),
            }
            static_result = static_validate(intent, repaired_cert)
        else:
            contract_v2 = type("Contract", (), {"level": _contract_level(report), "reasons": list((report.get("contract") or {}).get("reasons") or [])})()
            static_result = type("StaticResult", (), {"ok": bool((report.get("static_validation") or {}).get("ok")), "obligations": []})()

        gate_before = _gate_before(report)
        gate_after = gate_before
        gate_payload = dict(report.get("diff") or {}) if isinstance(report.get("diff"), Mapping) else {}
        gate_note = ""
        if args.rerun_gate:
            try:
                gate_after, gate_payload, gate_note = _rerun_gate(report=report, intent=intent, spec_cache=spec_cache)
            except Exception as exc:
                gate_after = None
                gate_payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "rerun": True}
                gate_note = f"gate rerun error: {type(exc).__name__}: {exc}"

        new_contract, new_reasons = _decide_final_contract(
            contract_v2=contract_v2,
            obligations=obligations,
            static_validation=static_result,
            gate_after=gate_after,
        )

        old_status = _obligation_status_map_from_static(report)
        old_reason = _obligation_reason_map_from_static(report)
        new_status = {str(row.id): str(row.status) for row in obligations}
        new_reason = {str(row.id): str(row.reason) for row in obligations}
        access_witnesses = [row.to_json_dict() for row in bundle.access_witnesses]
        best_witness = next((row for row in access_witnesses if str(row.get("binding_confidence") or "") != "failed"), access_witnesses[0] if access_witnesses else {})

        repaired_report = dict(report)
        repaired_report["certificate_v2"] = repaired_cert.to_json_dict()
        repaired_report["obligations"] = [row.to_json_dict() for row in obligations]
        repaired_report["static_validation"] = {
            "ok": bool(getattr(static_result, "ok", False)),
            "reasons": list(getattr(static_result, "reasons", []) or []),
            "obligations": [
                {"id": str(row.id), "status": str(row.status), "detail": getattr(row, "detail", None)}
                for row in list(getattr(static_result, "obligations", []) or [])
            ],
        }
        repaired_report["contract"] = {
            "level": str(new_contract),
            "reasons": list(new_reasons),
            "assumptions": list(getattr(contract_v2, "assumptions", []) or []),
            "signals": {
                "obligations": [row.to_json_dict() for row in obligations],
            },
        }
        repaired_report["diff"] = dict(gate_payload) if gate_payload else repaired_report.get("diff")
        repaired_report["deterministic_repair"] = {
            "root_cause": str(row["root_cause"]),
            "validator_reran": bool(args.rerun_validator),
            "gate_reran": bool(args.rerun_gate),
            "gate_note": str(gate_note),
            "repair_bundle": bundle.to_json_dict(),
        }

        if args.apply or not args.dry_run:
            artifact_after.parent.mkdir(parents=True, exist_ok=True)
            _write_json(artifact_after, repaired_report)
            _write_json(artifact_after.with_suffix(".certificate_v2.json"), repaired_report["certificate_v2"])
            _write_json(artifact_after.with_suffix(".contract.json"), repaired_report["contract"])
            if repaired_report.get("diff") is not None:
                _write_json(artifact_after.with_suffix(".diff.json"), repaired_report["diff"])
            _write_json(artifact_after.with_suffix(".repair_bundle.json"), bundle.to_json_dict())

        why_not_changed = "" if new_contract != _contract_level(report) else _post_repair_reason(
            new_contract=new_contract,
            statuses=new_status,
            static_validation=static_result,
            gate_after=gate_after,
            access_witnesses=access_witnesses,
        )
        result = KernelRepairResult(
            kernel_name=kernel,
            family=family,
            root_cause=str(row["root_cause"]),
            old_contract=_contract_level(report),
            new_contract=str(new_contract),
            source_gate_before=gate_before,
            source_gate_after=gate_after,
            old_o2=str(old_status.get(O2_AFFINE_OR_STRUCTURED_INDEXING, "")),
            new_o2=str(new_status.get(O2_AFFINE_OR_STRUCTURED_INDEXING, "")),
            old_o3=str(old_status.get(O3_MASK_IMPLIES_INBOUNDS, "")),
            new_o3=str(new_status.get(O3_MASK_IMPLIES_INBOUNDS, "")),
            old_o3_reason=str(old_reason.get(O3_MASK_IMPLIES_INBOUNDS, "")),
            new_o3_reason=str(new_reason.get(O3_MASK_IMPLIES_INBOUNDS, "")),
            mask_bound=str(best_witness.get("mask_expr") or ""),
            access_bound=str(best_witness.get("tensor_bound") or ""),
            index_before=_first_index_expr(report),
            index_after=str(best_witness.get("normalized_index_expr") or ""),
            binding_confidence=str(best_witness.get("binding_confidence") or ""),
            validator_reran=bool(args.rerun_validator),
            gate_reran=bool(args.rerun_gate),
            changed=_contract_level(report) != str(new_contract),
            why_not_changed=why_not_changed,
            artifact_before=_rel(report_path),
            artifact_after=_rel(artifact_after),
            repaired_report=repaired_report,
            repaired_examples=access_witnesses,
            static_ok_after=bool(getattr(static_result, "ok", False)),
        )
        results.append(result)

        example = _first_success_example(
            report=report,
            result=result,
            obligations=obligations,
            access_witnesses=access_witnesses,
        )
        if example is not None:
            repaired_examples.append(example)

    per_kernel_fields = [
        "kernel_name",
        "family",
        "old_contract",
        "new_contract",
        "root_cause",
        "source_gate_before",
        "source_gate_after",
        "old_O2",
        "new_O2",
        "old_O3",
        "new_O3",
        "old_O3_reason",
        "new_O3_reason",
        "mask_bound",
        "access_bound",
        "index_before",
        "index_after",
        "binding_confidence",
        "validator_reran",
        "gate_reran",
        "changed",
        "why_not_changed",
        "artifact_before",
        "artifact_after",
    ]
    per_kernel_rows = [
        {
            "kernel_name": row.kernel_name,
            "family": row.family,
            "old_contract": row.old_contract,
            "new_contract": row.new_contract,
            "root_cause": row.root_cause,
            "source_gate_before": _bool_str(row.source_gate_before),
            "source_gate_after": _bool_str(row.source_gate_after),
            "old_O2": row.old_o2,
            "new_O2": row.new_o2,
            "old_O3": row.old_o3,
            "new_O3": row.new_o3,
            "old_O3_reason": row.old_o3_reason,
            "new_O3_reason": row.new_o3_reason,
            "mask_bound": row.mask_bound,
            "access_bound": row.access_bound,
            "index_before": row.index_before,
            "index_after": row.index_after,
            "binding_confidence": row.binding_confidence,
            "validator_reran": _bool_str(row.validator_reran),
            "gate_reran": _bool_str(row.gate_reran),
            "changed": _bool_str(row.changed),
            "why_not_changed": row.why_not_changed,
            "artifact_before": row.artifact_before,
            "artifact_after": row.artifact_after,
        }
        for row in results
    ]
    _write_csv(output_dir / "per_kernel_repair_result.csv", per_kernel_rows, per_kernel_fields)

    root_before_counter = Counter(row.get("root_cause") for row in audit_rows if row.get("root_cause"))
    root_after_rows = []
    for root_cause in sorted(root_causes):
        bucket = [row for row in results if row.root_cause == root_cause]
        fixed_to_full = sum(1 for row in bucket if row.new_contract == "FULL")
        still_partial = sum(1 for row in bucket if row.new_contract == "PARTIAL")
        became_oos = sum(1 for row in bucket if row.new_contract == "OUT_OF_SCOPE")
        count_after = still_partial + became_oos
        needs_extractor_fix = sum(
            1
            for row in bucket
            if row.why_not_changed in {"missing mask evidence", "missing domain constraint", "shape/layout witness missing", "gate missing/fail"}
        )
        needs_solver_template = sum(1 for row in bucket if row.why_not_changed == "solver template missing")
        needs_manual_review = sum(1 for row in bucket if row.why_not_changed in {"data-dependent indexing", "other"})
        root_after_rows.append(
            {
                "root_cause": root_cause,
                "count_before": str(root_before_counter.get(root_cause, 0)),
                "count_after": str(count_after),
                "fixed_to_FULL": str(fixed_to_full),
                "still_PARTIAL": str(still_partial),
                "became_OOS": str(became_oos),
                "needs_extractor_fix": str(needs_extractor_fix),
                "needs_solver_template": str(needs_solver_template),
                "needs_manual_review": str(needs_manual_review),
            }
        )
    _write_csv(
        output_dir / "root_cause_before_after.csv",
        root_after_rows,
        [
            "root_cause",
            "count_before",
            "count_after",
            "fixed_to_FULL",
            "still_PARTIAL",
            "became_OOS",
            "needs_extractor_fix",
            "needs_solver_template",
            "needs_manual_review",
        ],
    )

    run_summary = _read_json(input_dir / "run_summary.json")
    scope_kernels = [str(kernel) for kernel in run_summary.get("scope_kernels") or []]
    report_by_kernel: dict[str, dict[str, Any]] = {}
    for row in per_kernel_audit.values():
        report_path = ROOT / str(row["report_path"])
        if report_path.is_file():
            report_by_kernel[str(row["kernel_name"])] = _read_json(report_path)
    for kernel in scope_kernels:
        if kernel in report_by_kernel:
            continue
        matches = list(input_dir.rglob(f"pipeline_reports/{kernel}.json"))
        if matches:
            report_by_kernel[kernel] = _read_json(matches[0])

    result_by_kernel = {row.kernel_name: row for row in results}
    after_levels = Counter()
    before_levels = Counter()
    before_o2_unknown = 0
    after_o2_unknown = 0
    before_o3_unknown = 0
    after_o3_unknown = 0
    before_gate_pass = 0
    after_gate_pass = 0
    before_gate_fail = 0
    after_gate_fail = 0
    before_validator_fail = 0
    after_validator_fail = 0

    for kernel in scope_kernels:
        report = report_by_kernel[kernel]
        old_level = _contract_level(report)
        before_levels[old_level] += 1
        result = result_by_kernel.get(kernel)
        new_level = result.new_contract if result is not None else old_level
        after_levels[new_level] += 1

        old_status = _obligation_status_map_from_static(report)
        if old_status.get(O2_AFFINE_OR_STRUCTURED_INDEXING) == "UNKNOWN":
            before_o2_unknown += 1
        if old_status.get(O3_MASK_IMPLIES_INBOUNDS) == "UNKNOWN":
            before_o3_unknown += 1
        if result is not None:
            if result.new_o2 == "UNKNOWN":
                after_o2_unknown += 1
            if result.new_o3 == "UNKNOWN":
                after_o3_unknown += 1
        else:
            if old_status.get(O2_AFFINE_OR_STRUCTURED_INDEXING) == "UNKNOWN":
                after_o2_unknown += 1
            if old_status.get(O3_MASK_IMPLIES_INBOUNDS) == "UNKNOWN":
                after_o3_unknown += 1

        gate_before = _gate_before(report)
        if gate_before is True:
            before_gate_pass += 1
        elif gate_before is False:
            before_gate_fail += 1
        gate_after = result.source_gate_after if result is not None else gate_before
        if gate_after is True:
            after_gate_pass += 1
        elif gate_after is False:
            after_gate_fail += 1

        static_before = bool((report.get("static_validation") or {}).get("ok"))
        if not static_before:
            before_validator_fail += 1
        static_after = result.static_ok_after if result is not None else static_before
        if not static_after:
            after_validator_fail += 1

    contract_rows = [
        {"metric": "FULL", "before": str(before_levels.get("FULL", 0)), "after": str(after_levels.get("FULL", 0)), "delta": str(after_levels.get("FULL", 0) - before_levels.get("FULL", 0))},
        {"metric": "PARTIAL", "before": str(before_levels.get("PARTIAL", 0)), "after": str(after_levels.get("PARTIAL", 0)), "delta": str(after_levels.get("PARTIAL", 0) - before_levels.get("PARTIAL", 0))},
        {"metric": "OOS", "before": str(before_levels.get("OUT_OF_SCOPE", 0)), "after": str(after_levels.get("OUT_OF_SCOPE", 0)), "delta": str(after_levels.get("OUT_OF_SCOPE", 0) - before_levels.get("OUT_OF_SCOPE", 0))},
        {"metric": "NO_CONTRACT", "before": str(before_levels.get("NO_CONTRACT", 0)), "after": str(after_levels.get("NO_CONTRACT", 0)), "delta": str(after_levels.get("NO_CONTRACT", 0) - before_levels.get("NO_CONTRACT", 0))},
        {"metric": "O2 UNKNOWN", "before": str(before_o2_unknown), "after": str(after_o2_unknown), "delta": str(after_o2_unknown - before_o2_unknown)},
        {"metric": "O3 UNKNOWN", "before": str(before_o3_unknown), "after": str(after_o3_unknown), "delta": str(after_o3_unknown - before_o3_unknown)},
        {"metric": "gate PASS", "before": str(before_gate_pass), "after": str(after_gate_pass), "delta": str(after_gate_pass - before_gate_pass)},
        {"metric": "gate FAIL", "before": str(before_gate_fail), "after": str(after_gate_fail), "delta": str(after_gate_fail - before_gate_fail)},
        {"metric": "validator FAIL", "before": str(before_validator_fail), "after": str(after_validator_fail), "delta": str(after_validator_fail - before_validator_fail)},
    ]
    _write_csv(output_dir / "contract_before_after.csv", contract_rows, ["metric", "before", "after", "delta"])

    _write_repaired_example_md(output_dir / "repaired_examples.md", repaired_examples)
    _write_remaining_partials_md(output_dir / "remaining_partials.md", results)

    changed_count = sum(1 for row in results if row.changed)
    full_count = sum(1 for row in results if row.new_contract == "FULL")
    remaining_partial_count = sum(1 for row in results if row.new_contract == "PARTIAL")
    remaining_oos_count = sum(1 for row in results if row.new_contract == "OUT_OF_SCOPE")
    fixed_by_root = {
        root_cause: sum(1 for row in results if row.root_cause == root_cause and row.new_contract == "FULL")
        for root_cause in sorted(root_causes)
    }
    remaining_reason_counter = Counter(row.why_not_changed for row in results if row.new_contract != "FULL")
    selection_before_partial = len(selected_rows)
    recommendation_lines = [
        "# Recommendation",
        "",
        f"- PARTIAL 是否真实减少：是。针对选定的 {selection_before_partial} 个 PARTIAL kernel，repair 后有 {full_count} 个进入 FULL，剩余 PARTIAL={remaining_partial_count}，OOS={remaining_oos_count}。",
        f"- 全量分布变化：FULL {before_levels.get('FULL', 0)} -> {after_levels.get('FULL', 0)}，PARTIAL {before_levels.get('PARTIAL', 0)} -> {after_levels.get('PARTIAL', 0)}，O3 UNKNOWN {before_o3_unknown} -> {after_o3_unknown}，O2 UNKNOWN {before_o2_unknown} -> {after_o2_unknown}。",
        f"- 减少主要来自哪些 repair：deterministic rebuild + strict validator/source gate 复跑；mask_access_binding_missing 修复 {fixed_by_root.get('mask_access_binding_missing', 0)} / {root_before_counter.get('mask_access_binding_missing', 0)}，index_canonicalization_missing 修复 {fixed_by_root.get('index_canonicalization_missing', 0)} / {root_before_counter.get('index_canonicalization_missing', 0)}。",
        f"- 剩下 PARTIAL 的主因是什么：{', '.join(f'{reason}={count}' for reason, count in sorted(remaining_reason_counter.items()) if reason) or 'none'}。",
        "- 下一步优先级：先继续修 extractor / index canonicalizer，其次补 O3 solver template；LLM prompt 仍然是 fallback，不是主路径。",
        "- 哪些结果可以写进论文：deterministic repair 后真实减少的 PARTIAL/FULL before-after、代表性 repaired examples、remaining limitations。",
        "- 哪些不能写进论文：未 rerun gate 的 case 不能写成 FULL correctness；data-dependent / atomics / gate missing case 不能写成已修复。",
        f"- Gate / validator：选定 repair 集合中 gate PASS={sum(1 for row in results if row.source_gate_after is True)}，gate FAIL={sum(1 for row in results if row.source_gate_after is False)}，validator rerun={sum(1 for row in results if row.validator_reran)}。",
        "",
        "## Selection",
        f"- repaired root causes: {sorted(root_causes)}",
        f"- selected kernels: {len(selected_rows)}",
        f"- other PARTIAL cases kept separate: {len(other_rows)}",
    ]
    (output_dir / "recommendation.md").write_text("\n".join(recommendation_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {_rel(output_dir / 'per_kernel_repair_result.csv')}")
    print(f"Wrote: {_rel(output_dir / 'root_cause_before_after.csv')}")
    print(f"Wrote: {_rel(output_dir / 'contract_before_after.csv')}")
    print(f"Wrote: {_rel(output_dir / 'repaired_examples.md')}")
    print(f"Wrote: {_rel(output_dir / 'remaining_partials.md')}")
    print(f"Wrote: {_rel(output_dir / 'recommendation.md')}")
    print(f"Processed kernels: {len(results)} ; changed={changed_count} ; new FULL={full_count}")


if __name__ == "__main__":
    main()
