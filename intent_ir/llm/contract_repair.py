"""
LLM prompt builder for mechanically-checkable contract repair candidates.

This module is intentionally separate from the normal IntentIR extraction hub:
- extraction returns CandidateIntent JSON
- contract repair returns a proof-witness / repair-candidate JSON blob

The repair agent focuses on obligation evidence (especially O2/O3) and must not
promote contracts by assertion.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .llm_extract import extract_json_object_with_trace
from .llm_client import DEFAULT_MODEL


SYSTEM_PROMPT = """You are an IntentIR contract-repair agent.

Your goal is to reduce PARTIAL contracts only by producing mechanically checkable evidence.
Do NOT promote a kernel to FULL by assertion. A kernel may become FULL only if every critical obligation can be rechecked by the validator.

Input:
1. Source kernel excerpt or normalized source IR.
2. KernelDescriptor.
3. CanonicalEvidence.
4. Current IntentIR JSON.
5. Current obligation report, especially O2/O3 UNKNOWN reasons.
6. Tensor shapes, symbolic domains, strides, masks, and access patterns if available.

Task:
Repair the IntentIR and evidence witnesses for O2/O3.

For every memory access:
- Identify the tensor name.
- Identify the index expression used by the load/store.
- Identify the exact mask/predicate guarding this access.
- Bind the mask to the access. Do not use a mask from another access.
- Normalize the index expression into one of:
  1D affine: base + stride * i
  multi-dimensional affine: row * stride0 + col * stride1 + ...
  flattened tensor index: row * N + col
  structured where/clamp/min/max form
  unsupported data-dependent form
- Extract all domain constraints:
  arange range
  program_id/block_id range
  symbolic shape positivity
  loop bounds
  broadcast dimension constraints
  stride/layout constraints
- Prove or provide a proof witness for:
  mask/access predicate => 0 <= index < tensor_bound
- If multi-dimensional, prove each axis bound first:
  0 <= row < M
  0 <= col < N
  then prove flattened bound:
  0 <= row * N + col < M * N

Rules:
- If a proof requires an assumption, list it explicitly in "assumptions".
- If evidence is missing from the source/evidence package, do not invent it.
- If the index is data-dependent, mark O2/O3 as unrepaired and explain why.
- If atomics are present and cannot be shown to be controlled reductions, keep the kernel OOS.
- Do not change Layer A operator semantics unless the obligation failure is caused by a clearly wrong semantic classification.
- Layer C hints must not be used to justify correctness.

Output JSON:
{
  "repair_status": "repaired" | "partially_repaired" | "unrepaired",
  "layer_a_changes": [...],
  "layer_b_changes": [...],
  "new_evidence_witnesses": [
    {
      "access_id": "...",
      "tensor": "...",
      "index_expr_original": "...",
      "index_expr_normalized": "...",
      "mask_expr": "...",
      "domain_constraints": [...],
      "bound": "...",
      "o2_result_expected": "PASS" | "UNKNOWN" | "FAIL",
      "o3_result_expected": "PASS" | "UNKNOWN" | "FAIL",
      "proof_sketch": "...",
      "required_validator_template": "1d_affine" | "2d_flattened" | "where_split" | "clamp" | "broadcast" | "unsupported"
    }
  ],
  "remaining_unknowns": [
    {
      "obligation": "O2" | "O3" | "O5" | "O7",
      "reason": "...",
      "missing_evidence": [...]
    }
  ],
  "final_contract_recommendation": "FULL" | "PARTIAL" | "OOS",
  "why_not_full_if_partial": "..."
}

Important:
The final validator, not you, decides the real contract. Your job is only to produce repair candidates and proof witnesses.

Return STRICT JSON only. Do not emit prose or code fences."""


OUTPUT_SCHEMA_HINT = {
    "repair_status": "repaired | partially_repaired | unrepaired",
    "layer_a_changes": [],
    "layer_b_changes": [],
    "new_evidence_witnesses": [
        {
            "access_id": "...",
            "tensor": "...",
            "index_expr_original": "...",
            "index_expr_normalized": "...",
            "mask_expr": "...",
            "domain_constraints": ["..."],
            "bound": "...",
            "o2_result_expected": "PASS | UNKNOWN | FAIL",
            "o3_result_expected": "PASS | UNKNOWN | FAIL",
            "proof_sketch": "...",
            "required_validator_template": "1d_affine | 2d_flattened | where_split | clamp | broadcast | unsupported",
        }
    ],
    "remaining_unknowns": [{"obligation": "O2 | O3 | O5 | O7", "reason": "...", "missing_evidence": ["..."]}],
    "final_contract_recommendation": "FULL | PARTIAL | OOS",
    "why_not_full_if_partial": "...",
}


def _truncate_text(text: Any, *, max_chars: int) -> str:
    raw = str(text or "")
    if max_chars <= 0 or len(raw) <= max_chars:
        return raw
    keep_head = max(256, int(max_chars * 0.8))
    keep_tail = max(128, int(max_chars * 0.15))
    clipped = raw[:keep_head].rstrip()
    suffix = raw[-keep_tail:].lstrip() if keep_tail < len(raw) else ""
    return (
        f"[IntentIR contract-repair] SOURCE TRUNCATED: original_chars={len(raw)} "
        f"kept_head={len(clipped)} kept_tail={len(suffix)}\n"
        f"{clipped}\n"
        f"[IntentIR contract-repair] ... TRUNCATED ...\n"
        f"{suffix}"
    ).strip()


def _as_dict(obj: Any) -> Dict[str, Any]:
    return dict(obj) if isinstance(obj, Mapping) else {}


def _pick_current_intent(report: Mapping[str, Any]) -> Dict[str, Any]:
    for key in ("intent_expanded", "intent", "intent_json"):
        value = report.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _pick_canonical_evidence(report: Mapping[str, Any]) -> Dict[str, Any]:
    cert_v2 = _as_dict(report.get("certificate_v2"))
    semantic_facts = _as_dict(cert_v2.get("semantic_facts"))
    canonical = semantic_facts.get("canonical_evidence")
    return dict(canonical) if isinstance(canonical, Mapping) else {}


def _pick_target_obligations(report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    contract = _as_dict(report.get("contract"))
    signals = _as_dict(contract.get("signals"))
    signal_obligations = list(signals.get("obligations") or [])
    static_validation = _as_dict(report.get("static_validation"))
    static_obligations = list(static_validation.get("obligations") or [])

    out: List[Dict[str, Any]] = []
    interesting_ids = {
        "O2_affine_or_structured_indexing": "O2",
        "O3_mask_implies_inbounds": "O3",
        "O5_no_data_dependent_address": "O5",
        "O7_no_atomics_or_controlled_atomics": "O7",
    }

    for raw in signal_obligations:
        if not isinstance(raw, Mapping):
            continue
        obligation_id = str(raw.get("id") or "").strip()
        short_id = interesting_ids.get(obligation_id)
        if short_id is None:
            continue
        status = str(raw.get("status") or "").strip()
        if status == "PASS":
            continue
        out.append(
            {
                "source": "contract.signals.obligations",
                "obligation_id": obligation_id,
                "obligation": short_id,
                "status": status,
                "reason": str(raw.get("reason") or ""),
                "witness": raw.get("witness"),
            }
        )

    for raw in static_obligations:
        if not isinstance(raw, Mapping):
            continue
        obligation_id = str(raw.get("id") or "").strip()
        short_id = interesting_ids.get(obligation_id)
        if short_id is None:
            continue
        status = str(raw.get("status") or "").strip()
        if status == "PASS":
            continue
        out.append(
            {
                "source": "static_validation.obligations",
                "obligation_id": obligation_id,
                "obligation": short_id,
                "status": status,
                "reason": str(raw.get("detail") or ""),
            }
        )
    return out


def build_repair_input_package(report: Mapping[str, Any], *, source_char_limit: int = 20000) -> Dict[str, Any]:
    descriptor = _as_dict(report.get("descriptor"))
    certificate = _as_dict(report.get("certificate"))
    certificate_v2 = _as_dict(report.get("certificate_v2"))
    contract = _as_dict(report.get("contract"))
    static_validation = _as_dict(report.get("static_validation"))
    current_intent = _pick_current_intent(report)
    canonical_evidence = _pick_canonical_evidence(report)
    descriptor_meta = _as_dict(descriptor.get("meta"))
    descriptor_launch = _as_dict(descriptor.get("launch"))

    source_text = _truncate_text(descriptor.get("source_text"), max_chars=int(source_char_limit))
    baseline = _as_dict(report.get("baseline"))
    cases = _as_dict(report.get("cases"))
    verification_config = _as_dict(report.get("verification_config"))
    report_meta = {
        "kernel": str(report.get("kernel") or descriptor.get("name") or ""),
        "frontend": str(descriptor.get("frontend") or ""),
        "backend_target": str(report.get("backend_target") or descriptor_meta.get("backend_target") or ""),
        "source_op": str(descriptor_meta.get("source_op") or ""),
        "provider": str(descriptor_meta.get("triton_provider") or ""),
        "artifact_dir": str(descriptor_meta.get("artifact_dir") or ""),
    }

    package = {
        "source_kernel_excerpt": source_text,
        "normalized_source_ir_hint": {
            "ttir_path": str(_as_dict(descriptor.get("artifacts")).get("ttir_path") or ""),
            "ttir_text_present": bool(_as_dict(descriptor.get("artifacts")).get("ttir_text")),
        },
        "kernel_descriptor": descriptor,
        "canonical_evidence": canonical_evidence,
        "current_intentir_json": current_intent,
        "current_contract": contract,
        "current_obligation_report": {
            "target_obligations": _pick_target_obligations(report),
            "contract_signals_obligations": list(_as_dict(contract.get("signals")).get("obligations") or []),
            "static_validation_obligations": list(static_validation.get("obligations") or []),
            "static_validation_ok": static_validation.get("ok"),
            "static_validation_reasons": list(static_validation.get("reasons") or []),
        },
        "shape_and_access_context": {
            "canonical_shapes": _as_dict(descriptor_launch.get("canonical_shapes")),
            "baseline_shapes": _as_dict(baseline.get("shapes")),
            "in_contract_cases": list(cases.get("in_contract") or []),
            "out_of_contract_cases": list(cases.get("out_of_contract") or []),
            "intent_schedule_hints_v2": _as_dict(_as_dict(current_intent.get("meta")).get("schedule_hints_v2")),
            "intent_access_witness": _as_dict(_as_dict(current_intent.get("meta")).get("access_witness")),
            "certificate_v2_schedule_hints": _as_dict(certificate_v2.get("schedule_hints")),
            "legacy_mask_constraints": _as_dict(certificate.get("mask_constraints")),
            "legacy_mask_formulas": _as_dict(certificate.get("mask_formulas")),
            "legacy_mask_accesses": _as_dict(certificate.get("mask_accesses")),
            "legacy_pointer_groups": _as_dict(certificate.get("pointer_groups")),
        },
        "verification_context": verification_config,
        "guardrails": {
            "forbid_layer_c_correctness_justification": True,
            "validator_is_final_decider": True,
            "only_mechanically_checkable_evidence": True,
        },
        "report_meta": report_meta,
    }
    return package


def build_messages(
    repair_input: Mapping[str, Any],
    *,
    extra_instruction: Optional[str] = None,
) -> List[Dict[str, str]]:
    payload = json.dumps(dict(repair_input), ensure_ascii=False, sort_keys=True, indent=2)
    user_lines = [
        "Input package (JSON):",
        payload,
        "",
        "Focus on O2/O3 UNKNOWN or FAIL first, but keep O5/O7 accurate when they block promotion.",
        "Never invent masks, domains, or layout facts that are absent from the package.",
        "",
        "Output schema reminder:",
        json.dumps(OUTPUT_SCHEMA_HINT, ensure_ascii=False, indent=2),
    ]
    if extra_instruction:
        user_lines += ["", "Extra instruction:", str(extra_instruction)]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


def extract_repair_json(
    repair_input: Mapping[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    extra_instruction: Optional[str] = None,
    max_parse_retries: int = 2,
    **chat_kwargs: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, str]]]:
    messages = build_messages(repair_input, extra_instruction=extra_instruction)
    payload, trace = extract_json_object_with_trace(
        messages,
        model=model,
        max_parse_retries=max_parse_retries,
        **chat_kwargs,
    )
    return payload, trace, messages


__all__ = [
    "SYSTEM_PROMPT",
    "OUTPUT_SCHEMA_HINT",
    "build_repair_input_package",
    "build_messages",
    "extract_repair_json",
]
