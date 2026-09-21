"""Capability matching: does a tool for this hypothesis already exist?

Run strictly after blind hypothesis generation, and given only the hypotheses and
the operation catalogue -- never the anomaly, never the freedom to restate a
hypothesis. Its single job is a yes/no per hypothesis:

* ``matched`` -- an existing operation achieves the WHOLE required change, with
  parameters that follow from the hypothesis;
* ``gap``     -- it does not, and a new operation has to be written.

The bar for ``matched`` is deliberately high. A partial match is a gap: an
operation that achieves half of what a hypothesis asks for, applied alone, gets a
verdict on a question nobody posed. Reporting a gap is the useful answer, not a
failure, so the prompt says so explicitly -- otherwise a helpful model rounds
every hypothesis to the nearest available tool, which is exactly the bias the blind
stage exists to remove.
"""
from __future__ import annotations

import json

from project.ato.grammar import all_operations, validate_candidate

MATCH_SYSTEM_PROMPT = """You map hypotheses about a flawed engineering problem statement onto \
the edit operations a piece of software can actually perform.

For each hypothesis, decide ONE of:

  matched -- an existing operation performs the ENTIRE change the hypothesis requires, and the \
hypothesis determines its parameters. Give the operation name and parameter values.

  gap     -- no existing operation performs that change, or one performs only part of it, or \
performing it would require reinterpreting the hypothesis into something weaker.

RULES

1. A PARTIAL match is a gap. Do not stretch an operation to cover a hypothesis it only \
half-addresses. If the hypothesis asks for something in the analysis model and the closest \
operation changes a mesh parameter, that is a gap, not a match.

2. Reporting a gap is a correct and useful answer. Missing capability will be BUILT, so there \
is no penalty for saying it is missing. There is a large penalty for a false match: it puts a \
verdict on a question that was never asked.

3. Do not invent operation names. If you are naming something that is not in the catalogue \
below, the answer is gap.

4. Parameters must be values the hypothesis or the specification determines. If the hypothesis \
says the value is unknown, still choose a defensible probe value and say in `note` that it is a \
probe, not a determination."""

MATCH_TOOL = {
    "name": "match_capabilities",
    "description": "For each hypothesis, report whether an existing operation covers it.",
    "input_schema": {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "hypothesis_id": {"type": "string"},
                        "decision": {"type": "string", "enum": ["matched", "gap"]},
                        "operation": {"type": "string",
                                      "description": "catalogue operation name, only when matched"},
                        "parameters": {"type": "object"},
                        "missing_capability": {
                            "type": "string",
                            "description": ("when gap: what the new operation must be able to do, "
                                            "stated as a capability rather than as code")},
                        "note": {"type": "string"},
                    },
                    "required": ["hypothesis_id", "decision", "note"],
                },
            }
        },
        "required": ["decisions"],
    },
}


def match(backend, hypotheses: list) -> dict:
    """Return ``{"matched": [candidate...], "gaps": [(hypothesis, missing)...]}``."""
    catalogue = [op.to_prompt_dict() for op in all_operations().values()]
    user = (
        "AVAILABLE OPERATIONS (the complete catalogue; nothing else exists):\n"
        + json.dumps(catalogue, indent=2)
        + "\n\nHYPOTHESES to map:\n"
        + json.dumps(hypotheses, indent=2, default=str)
    )
    raw = backend(MATCH_SYSTEM_PROMPT, user, MATCH_TOOL)
    by_id = {h["id"]: h for h in hypotheses}
    ops = all_operations()

    matched, gaps, rejected = [], [], []
    seen = set()
    for d in raw.get("decisions", []):
        hid = d.get("hypothesis_id")
        h = by_id.get(hid)
        if h is None:
            rejected.append({"decision": d, "errors": ["unknown hypothesis id"]})
            continue
        seen.add(hid)
        if d.get("decision") == "matched":
            name = d.get("operation")
            if name not in ops:
                gaps.append({"hypothesis": h,
                             "missing_capability": f"claimed operation {name!r} does not exist",
                             "note": d.get("note", "")})
                continue
            cand = {"id": hid, "operation": name, "parameters": dict(d.get("parameters") or {}),
                    "targets_anomaly": h.get("targets_anomaly", []),
                    "rationale": h.get("rationale", ""),
                    "proposed_tier": h.get("proposed_tier"),
                    "origin": "committed_grammar", "hypothesis": h}
            errs = validate_candidate(cand)
            if errs:
                # A claimed match that fails validation is not a dead end: the
                # hypothesis is still live and no tool covers it, which is exactly a
                # gap. Dropping it here would silently lose a hypothesis.
                rejected.append({"candidate": cand, "errors": errs})
                gaps.append({"hypothesis": h,
                             "missing_capability": (
                                 f"the closest existing operation {name!r} does not accept the "
                                 f"parameters this hypothesis implies ({'; '.join(errs)}), so it "
                                 f"does not in fact cover it"),
                             "note": d.get("note", "")})
                continue
            matched.append(cand)
        else:
            gaps.append({"hypothesis": h,
                         "missing_capability": d.get("missing_capability")
                         or h.get("required_change", ""),
                         "note": d.get("note", "")})
    for hid, h in by_id.items():
        if hid not in seen:
            gaps.append({"hypothesis": h,
                         "missing_capability": h.get("required_change", ""),
                         "note": "matcher returned no decision for this hypothesis"})
    return {"matched": matched, "gaps": gaps, "rejected": rejected,
            "model": getattr(backend, "model", "unknown")}
