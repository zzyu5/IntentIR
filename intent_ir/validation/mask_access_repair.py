from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from frontends.common.certificate_v2 import SemanticCertificateV2
from frontends.common.evidence import AccessSummary, CanonicalEvidence, IndexExpr, Predicate
from frontends.common.smt_o3 import _node_to_affine, _parse_cmp, _parse_int_expr  # type: ignore[attr-defined]
from frontends.triton.affine_expr import affine_from_ssa, build_aliases, expr_to_str, format_affine
from frontends.triton.certificate import build_certificate_v2
from frontends.triton.facts import AccessSite, TTIRFacts, extract_facts
from frontends.triton.ttir_witness import parse_function_args, parse_ssa_defs, trace_base_pointer
from pipeline.interfaces import KernelArtifactBundle, KernelDescriptor


_SSA_TOKEN_RE = re.compile(r"(%[A-Za-z0-9_]+)")
_UPPER_OPS = {"<", "<=", ">", ">="}


@dataclass(frozen=True)
class AccessRepairWitness:
    access_id: str
    tensor: str
    access_kind: str
    pointer_expr: str
    index_expr: str
    normalized_index_expr: str
    mask_expr: str
    tensor_bound: str
    domain_constraints: list[str] = field(default_factory=list)
    source_span: str = ""
    binding_confidence: str = "failed"
    binding_reason: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "access_id": str(self.access_id),
            "tensor": str(self.tensor),
            "access_kind": str(self.access_kind),
            "pointer_expr": str(self.pointer_expr),
            "index_expr": str(self.index_expr),
            "normalized_index_expr": str(self.normalized_index_expr),
            "mask_expr": str(self.mask_expr),
            "tensor_bound": str(self.tensor_bound),
            "domain_constraints": list(self.domain_constraints),
            "source_span": str(self.source_span),
            "binding_confidence": str(self.binding_confidence),
            "binding_reason": str(self.binding_reason),
        }


@dataclass(frozen=True)
class RepairCertificateBundle:
    descriptor: KernelDescriptor
    base_certificate_v2: SemanticCertificateV2
    repaired_certificate_v2: SemanticCertificateV2
    access_witnesses: list[AccessRepairWitness]
    changed_access_count: int
    repair_notes: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "descriptor": self.descriptor.to_json_dict(),
            "base_certificate_v2": self.base_certificate_v2.to_json_dict(),
            "repaired_certificate_v2": self.repaired_certificate_v2.to_json_dict(),
            "access_witnesses": [row.to_json_dict() for row in self.access_witnesses],
            "changed_access_count": int(self.changed_access_count),
            "repair_notes": list(self.repair_notes),
        }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _descriptor_from_report(report: Mapping[str, Any]) -> KernelDescriptor:
    raw = dict(report.get("descriptor") or {})
    bundle = KernelArtifactBundle(**dict(raw.get("artifacts") or {}))
    desc = KernelDescriptor(**{k: v for k, v in raw.items() if k != "artifacts"})
    desc.artifacts = bundle
    return desc


def _read_ttir(desc: KernelDescriptor) -> str:
    if desc.artifacts.ttir_text:
        return str(desc.artifacts.ttir_text)
    if desc.artifacts.ttir_path:
        path = Path(str(desc.artifacts.ttir_path))
        if path.is_file():
            return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"ttir not found for kernel {desc.name}")


def _base_arg_to_tensor(base_arg: str, desc: KernelDescriptor) -> str:
    arg_names = desc.io_spec.get("arg_names")
    if isinstance(arg_names, list):
        m = re.fullmatch(r"%arg(\d+)", str(base_arg))
        if m is not None:
            idx = int(m.group(1))
            if 0 <= idx < len(arg_names):
                name = arg_names[idx]
                if isinstance(name, str) and name:
                    return str(name)
    return str(base_arg[1:] if str(base_arg).startswith("%") else base_arg)


def _normalize_index_expr_text(index_exprs: Sequence[IndexExpr], *, address_index_exprs: Sequence[IndexExpr]) -> str:
    if len(index_exprs) == 1:
        return _format_ix(index_exprs[0])
    if index_exprs:
        axes = ", ".join(_format_ix(ix) for ix in index_exprs)
        if address_index_exprs:
            return f"axes=[{axes}] ; address={_format_ix(address_index_exprs[0])}"
        return f"axes=[{axes}]"
    if address_index_exprs:
        return _format_ix(address_index_exprs[0])
    return ""


def _format_ix(ix: IndexExpr) -> str:
    parts: list[str] = []
    for var, coeff in sorted((ix.terms or {}).items()):
        c = int(coeff)
        if c == 1:
            parts.append(str(var))
        elif c == -1:
            parts.append(f"-{var}")
        else:
            parts.append(f"{c}*{var}")
    if int(ix.const) != 0 or not parts:
        parts.append(str(int(ix.const)))
    return " + ".join(parts).replace("+ -", "- ")


def _domain_constraints(
    *,
    desc: KernelDescriptor,
    report: Mapping[str, Any],
    base_certificate_v2: SemanticCertificateV2,
) -> list[str]:
    out: list[str] = ["pid0 >= 0", "pid1 >= 0", "pid2 >= 0"]
    symbol_ranges = dict((base_certificate_v2.schedule_hints or {}).get("symbol_ranges") or {})
    for name, spec in sorted(symbol_ranges.items()):
        if isinstance(spec, Mapping):
            start = spec.get("start")
            end = spec.get("end")
            if isinstance(start, int) and isinstance(end, int):
                out.append(f"{name} in [{start}, {end})")
    canonical_shapes = dict((desc.launch or {}).get("canonical_shapes") or {})
    for name, value in sorted(canonical_shapes.items()):
        if isinstance(value, (int, float)) and int(value) > 0:
            out.append(f"{name} = {int(value)} > 0")
    baseline_shapes = dict((report.get("baseline") or {}).get("shapes") or {})
    for name, value in sorted(baseline_shapes.items()):
        if isinstance(value, (int, float)) and int(value) > 0 and f"{name} = {int(value)} > 0" not in out:
            out.append(f"{name} = {int(value)} > 0")
    legacy_index_symbols = dict((report.get("certificate") or {}).get("index_symbols") or {})
    ranges = dict(legacy_index_symbols.get("ranges") or {})
    for name, spec in sorted(ranges.items()):
        if isinstance(spec, Mapping):
            start = spec.get("start")
            end = spec.get("end")
            if isinstance(start, int) and isinstance(end, int):
                entry = f"{name} in [{start}, {end})"
                if entry not in out:
                    out.append(entry)
    return out


def _formula_has_or(mask_formula: str) -> bool:
    text = str(mask_formula or "").lower()
    return " or " in text or "arith.ori" in text or text.startswith("(") and ") or (" in text


def _legacy_mask_candidates(
    site: AccessSite,
    *,
    legacy_mask_accesses: Mapping[str, Any],
    legacy_mask_constraints: Mapping[str, Any],
) -> tuple[list[str], str, str]:
    mask = str(site.mask or "")
    ptr = str(site.ptr or "")
    exact: list[str] = []
    structural: list[str] = []
    heuristic: list[str] = []

    if mask and mask in legacy_mask_constraints:
        entries = legacy_mask_accesses.get(mask) or []
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, Mapping):
                continue
            if str(entry.get("kind") or "") == str(site.kind) and int(entry.get("line") or -1) == int(site.line_no) and str(entry.get("ptr") or "") == ptr:
                exact.append(mask)
                break
        if not exact:
            structural.append(mask)

    for candidate, entries in legacy_mask_accesses.items():
        if candidate == mask:
            continue
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            if str(entry.get("kind") or "") != str(site.kind):
                continue
            if int(entry.get("line") or -1) == int(site.line_no) and str(entry.get("ptr") or "") == ptr:
                exact.append(str(candidate))
                break
            if str(entry.get("ptr") or "") == ptr:
                structural.append(str(candidate))
            elif int(entry.get("line") or -1) == int(site.line_no):
                heuristic.append(str(candidate))

    if exact:
        return sorted(set(exact)), "exact", f"matched legacy mask by kind/line/ptr for {site.kind}@L{site.line_no}"
    if structural:
        return sorted(set(structural)), "structural", f"matched legacy mask structurally for {site.kind}@L{site.line_no}"
    if heuristic:
        return sorted(set(heuristic)), "heuristic", f"matched legacy mask heuristically for {site.kind}@L{site.line_no}"
    return [], "failed", f"no unique legacy mask for {site.kind}@L{site.line_no}"


def _clause_to_index_expr(clause: str) -> tuple[IndexExpr | None, IndexExpr | None, str]:
    try:
        cmp_ = _parse_cmp(str(clause))
        if cmp_ is None:
            return None, None, ""
        lhs_s, op, rhs_s = cmp_
        lhs_node = _parse_int_expr(lhs_s)
        rhs_node = _parse_int_expr(rhs_s)
        lhs_ix = _node_to_affine(lhs_node)
        rhs_ix = _node_to_affine(rhs_node)
        return lhs_ix, rhs_ix, op
    except Exception:
        return None, None, ""


def _sanitize_symbol_name(token: str) -> str:
    text = str(token or "").strip()
    if text.startswith("%arg"):
        return "arg" + text[len("%arg") :]
    if text.startswith("%"):
        return "ssa_" + text[1:]
    return text


def _normalize_symbolic_text(text: str) -> str:
    return _SSA_TOKEN_RE.sub(lambda m: _sanitize_symbol_name(m.group(1)), str(text or ""))


def _normalize_ssa_token(token: str, *, defs: Mapping[str, Any], aliases: Mapping[str, str]) -> str:
    symbol = str(token or "")
    aff = affine_from_ssa(symbol, defs, aliases=aliases)
    if not aff.non_affine:
        return format_affine(aff)
    pretty = expr_to_str(symbol, defs, aliases=aliases)
    return _normalize_symbolic_text(pretty)


def _normalize_clause_text(clause: str, *, defs: Mapping[str, Any], aliases: Mapping[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        token = str(match.group(1) or "")
        return f"({_normalize_ssa_token(token, defs=defs, aliases=aliases)})"

    text = _SSA_TOKEN_RE.sub(repl, str(clause or ""))
    text = _normalize_symbolic_text(text)
    return re.sub(r"\s+", " ", text).strip()


def _bound_like(ix: IndexExpr | None) -> bool:
    if ix is None:
        return False
    if not ix.terms:
        return True
    keys = list(ix.terms.keys())
    return all(str(key).startswith("%arg") or str(key).startswith("arg") or re.fullmatch(r"[A-Z][A-Z0-9_]*", str(key)) for key in keys)


def _index_side(ix: IndexExpr | None, other: IndexExpr | None) -> bool:
    if ix is None:
        return False
    if other is None:
        return True
    if _bound_like(other) and not _bound_like(ix):
        return True
    if _bound_like(ix) and not _bound_like(other):
        return False
    score = sum(1 for key in ix.terms.keys() if str(key).startswith("pid") or str(key).startswith("r") or str(key).startswith("%"))
    other_score = sum(1 for key in other.terms.keys() if str(key).startswith("pid") or str(key).startswith("r") or str(key).startswith("%"))
    return score >= other_score


def _logical_index_exprs_from_clauses(clauses: Sequence[str]) -> list[IndexExpr]:
    out: list[IndexExpr] = []
    seen: set[tuple[tuple[tuple[str, int], ...], int]] = set()
    for clause in clauses:
        lhs_ix, rhs_ix, _op = _clause_to_index_expr(str(clause))
        candidate = lhs_ix if _index_side(lhs_ix, rhs_ix) else rhs_ix
        if candidate is None:
            continue
        key = candidate.sort_key()
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def _tensor_bound_from_clauses(clauses: Sequence[str]) -> str:
    bounds: list[str] = []
    for clause in clauses:
        parsed = _parse_cmp(str(clause))
        if parsed is None:
            continue
        lhs, op, rhs = parsed
        if op not in _UPPER_OPS:
            continue
        lhs_ix, rhs_ix, _ = _clause_to_index_expr(str(clause))
        if lhs_ix is not None and _index_side(lhs_ix, rhs_ix):
            bounds.append(str(rhs))
        elif rhs_ix is not None:
            bounds.append(str(lhs))
    uniq = []
    seen: set[str] = set()
    for bound in bounds:
        if bound not in seen:
            uniq.append(bound)
            seen.add(bound)
    return " | ".join(uniq)


def _match_base_accesses(base_accesses: Sequence[AccessSummary]) -> dict[tuple[str, str], deque[AccessSummary]]:
    groups: dict[tuple[str, str], deque[AccessSummary]] = defaultdict(deque)
    for access in base_accesses:
        groups[(str(access.kind), str(access.tensor))].append(access)
    return groups


def _site_address_index(
    site: AccessSite,
    *,
    defs: Mapping[str, Any],
    func_args: set[str],
    aliases: Mapping[str, str],
) -> tuple[IndexExpr, bool]:
    ptr = str(site.ptr or "")
    if not ptr:
        return IndexExpr(terms={}, const=0), True
    base, offsets = _trace_addptr_chain(ptr, defs=defs, func_args=func_args)
    if base is None:
        return IndexExpr(terms={}, const=0), True
    total = None
    unresolved = False
    for off in offsets:
        aff = affine_from_ssa(off, defs, aliases=aliases)
        if aff.non_affine:
            unresolved = True
        if total is None:
            total = aff
        else:
            total = total.add(aff)
    if total is None:
        return IndexExpr(terms={}, const=0), True
    terms: dict[str, int] = {}
    for raw_var, coeff in total.coeff.items():
        if raw_var.startswith("%") and not raw_var.startswith("%arg"):
            unresolved = True
        terms[str(raw_var)] = int(coeff)
    terms = {k: v for k, v in terms.items() if v != 0}
    if total.non_affine:
        return IndexExpr(terms={}, const=0), True
    return IndexExpr(terms=terms, const=int(total.const)), unresolved


def _trace_addptr_chain(ptr: str, *, defs: Mapping[str, Any], func_args: set[str]) -> tuple[str | None, list[str]]:
    cur = str(ptr)
    offsets: list[str] = []
    for _ in range(64):
        if cur in func_args:
            return cur, list(reversed(offsets))
        d = defs.get(cur)
        if d is None:
            return None, []
        if getattr(d, "op", None) == "scf.iter_arg" and getattr(d, "operands", None):
            cur = d.operands[0]
            continue
        if getattr(d, "op", None) in {"tt.bitcast", "tt.broadcast", "tt.splat", "tt.reshape", "tt.expand_dims"} and getattr(d, "operands", None):
            cur = d.operands[0]
            continue
        if getattr(d, "op", None) == "tt.addptr" and len(getattr(d, "operands", ()) or ()) >= 2:
            offsets.append(d.operands[1])
            cur = d.operands[0]
            continue
        return None, []
    return None, []


def rebuild_repaired_certificate_bundle(report: Mapping[str, Any]) -> RepairCertificateBundle:
    desc = _descriptor_from_report(report)
    ttir = _read_ttir(desc)
    facts: TTIRFacts = extract_facts(ttir)
    defs = parse_ssa_defs(ttir)
    aliases = build_aliases(defs)
    func_args = set(parse_function_args(ttir).keys())

    base_certificate_v2 = build_certificate_v2(ttir, desc=desc, facts=facts)
    base_ce = base_certificate_v2.semantic_facts.get("canonical_evidence")
    base_accesses = list(base_ce.accesses) if isinstance(base_ce, CanonicalEvidence) else []
    base_groups = _match_base_accesses(base_accesses)

    legacy_certificate = dict(report.get("certificate") or {})
    legacy_mask_constraints = dict(legacy_certificate.get("mask_constraints") or {})
    legacy_mask_accesses = dict(legacy_certificate.get("mask_accesses") or {})
    legacy_mask_formulas = dict(legacy_certificate.get("mask_formulas") or {})

    shared_domains = _domain_constraints(desc=desc, report=report, base_certificate_v2=base_certificate_v2)

    repaired_accesses: list[AccessSummary] = []
    access_witnesses: list[AccessRepairWitness] = []
    changed_access_count = 0
    notes: list[str] = []
    exact_bind_count = 0
    fallback_bind_count = 0
    failed_bind_count = 0

    for site in list(facts.load_sites) + list(facts.store_sites):
        base_arg = trace_base_pointer(str(site.ptr or ""), defs, func_args) or ""
        tensor = _base_arg_to_tensor(base_arg, desc) if base_arg else (str(site.tensor_hint or "") or f"{site.kind}_{site.line_no}")
        base_queue = base_groups.get((str(site.kind), tensor))
        base_access = base_queue.popleft() if base_queue else None
        base_address_ixs = []
        base_meta = {}
        if base_access is not None:
            base_meta = dict(base_access.meta or {})
            raw_address = base_meta.get("address_index_exprs")
            if isinstance(raw_address, list):
                base_address_ixs = [ix for ix in raw_address if isinstance(ix, IndexExpr)]
        if not base_address_ixs:
            site_address_ix, site_unresolved = _site_address_index(site, defs=defs, func_args=func_args, aliases=aliases)
            base_address_ixs = [site_address_ix]
            base_meta = dict(base_meta)
            if site_unresolved:
                base_meta["unresolved"] = True

        candidates, confidence, reason = _legacy_mask_candidates(
            site,
            legacy_mask_accesses=legacy_mask_accesses,
            legacy_mask_constraints=legacy_mask_constraints,
        )
        chosen_mask = ""
        normalized_clauses: list[str] = []
        if len(candidates) == 1:
            chosen_mask = str(candidates[0])
            raw_formula = str(legacy_mask_formulas.get(chosen_mask) or "")
            if raw_formula and _formula_has_or(raw_formula):
                reason = f"{reason}; skipped because mask formula contains OR"
                confidence = "failed"
            else:
                normalized_clauses = [
                    _normalize_clause_text(str(clause), defs=defs, aliases=aliases)
                    for clause in list(legacy_mask_constraints.get(chosen_mask) or [])
                    if str(clause).strip()
                ]
        elif len(candidates) > 1:
            confidence = "failed"
            reason = f"ambiguous legacy mask candidates={candidates}"

        logical_index_exprs = _logical_index_exprs_from_clauses(normalized_clauses) if normalized_clauses else []
        predicate = base_access.predicate if base_access is not None else None
        index_exprs = list(base_access.index_exprs) if base_access is not None else list(base_address_ixs)
        binding_confidence = "failed"
        binding_reason = f"no recovered predicate for {site.kind}@L{site.line_no}"
        used_legacy_fallback = False

        if predicate is not None and list(predicate.clauses):
            if bool(site.has_mask):
                binding_confidence = "exact"
                binding_reason = f"rebuilt TTIR predicate for {site.kind}@L{site.line_no}"
            else:
                binding_confidence = "exact"
                binding_reason = f"mask-free access recovered directly from TTIR for {site.kind}@L{site.line_no}"
        elif not bool(site.has_mask):
            binding_confidence = "exact"
            binding_reason = f"mask-free access recovered directly from TTIR for {site.kind}@L{site.line_no}"

        if (
            confidence != "failed"
            and normalized_clauses
            and (
                predicate is None
                or not list(predicate.clauses)
                or (isinstance(base_meta, Mapping) and bool(base_meta.get("unresolved")))
            )
        ):
            predicate = Predicate(clauses=list(normalized_clauses))
            if logical_index_exprs:
                index_exprs = list(logical_index_exprs)
            binding_confidence = str(confidence)
            binding_reason = f"{reason}; fallback to legacy mask constraints"
            used_legacy_fallback = True
            changed_access_count += 1

        if not index_exprs:
            index_exprs = list(base_address_ixs)
        if not index_exprs:
            index_exprs = [IndexExpr(terms={}, const=0)]

        meta = dict(base_meta)
        meta["access_id"] = f"{site.kind}@L{int(site.line_no)}"
        meta["source_span"] = f"ttir:L{int(site.line_no)}"
        meta["pointer_expr"] = str(site.ptr or "")
        meta["binding_confidence"] = str(binding_confidence)
        meta["binding_reason"] = str(binding_reason)
        meta["mask_id"] = str(chosen_mask or site.mask or "")
        if used_legacy_fallback:
            meta.pop("unresolved", None)
        if binding_confidence == "exact":
            exact_bind_count += 1
        elif binding_confidence in {"structural", "heuristic"}:
            fallback_bind_count += 1
        else:
            failed_bind_count += 1

        repaired_access = AccessSummary(
            kind=str(site.kind),
            tensor=tensor,
            dtype=(str(base_access.dtype) if base_access is not None else "unknown"),
            rank=max(1, len(index_exprs)),
            index_exprs=list(index_exprs),
            predicate=predicate,
            address_space=(base_access.address_space if base_access is not None else "global"),
            meta=meta,
        )
        repaired_accesses.append(repaired_access)

        witness_clauses = list(predicate.clauses) if predicate is not None else []

        witness = AccessRepairWitness(
            access_id=str(meta["access_id"]),
            tensor=tensor,
            access_kind=str(site.kind),
            pointer_expr=str(site.ptr or ""),
            index_expr=(_format_ix(index_exprs[0]) if len(index_exprs) == 1 else " ; ".join(_format_ix(ix) for ix in index_exprs)),
            normalized_index_expr=_normalize_index_expr_text(index_exprs, address_index_exprs=base_address_ixs),
            mask_expr=(" && ".join(witness_clauses) if witness_clauses else ""),
            tensor_bound=_tensor_bound_from_clauses(witness_clauses),
            domain_constraints=list(shared_domains),
            source_span=f"ttir:L{int(site.line_no)}",
            binding_confidence=str(binding_confidence),
            binding_reason=str(binding_reason),
        )
        access_witnesses.append(witness)

    repaired_evidence = CanonicalEvidence(
        anchors=dict((base_certificate_v2.semantic_facts or {}).get("anchors") or {}),
        accesses=repaired_accesses,
        schedule_hints=dict(getattr(base_ce, "schedule_hints", {}) or {}) if isinstance(base_ce, CanonicalEvidence) else {},
        meta=dict(getattr(base_ce, "meta", {}) or {}) if isinstance(base_ce, CanonicalEvidence) else {},
    ).canonicalize()
    repaired_certificate_v2 = SemanticCertificateV2(
        schema_version=str(base_certificate_v2.schema_version),
        semantic_facts=dict(base_certificate_v2.semantic_facts),
        schedule_hints=dict(base_certificate_v2.schedule_hints or {}),
        meta=dict(base_certificate_v2.meta or {}),
    )
    repaired_certificate_v2.semantic_facts["canonical_evidence"] = repaired_evidence
    repaired_certificate_v2.meta["deterministic_repair"] = {
        "changed_access_count": int(changed_access_count),
        "access_witnesses": [row.to_json_dict() for row in access_witnesses],
    }
    repaired_certificate_v2.canonicalize()

    if changed_access_count > 0:
        notes.append(f"updated predicates/index witnesses for {changed_access_count} access(es)")
    else:
        notes.append("reused rebuilt certificate_v2 predicates/index witnesses without fallback edits")
    notes.append(f"binding summary: exact={exact_bind_count}, fallback={fallback_bind_count}, failed={failed_bind_count}")

    return RepairCertificateBundle(
        descriptor=desc,
        base_certificate_v2=base_certificate_v2,
        repaired_certificate_v2=repaired_certificate_v2,
        access_witnesses=access_witnesses,
        changed_access_count=changed_access_count,
        repair_notes=notes,
    )


__all__ = [
    "AccessRepairWitness",
    "RepairCertificateBundle",
    "rebuild_repaired_certificate_bundle",
]
