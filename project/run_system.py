"""Interactive natural-language runner for the verified 2-D SIMP pipeline.

Run from the repository root:

    /dolfinx-env/bin/python -m project.run_system
"""

from __future__ import annotations

import json
import math
import re


from project.parser.client import parse_problem
from project.parser.provenance import summarize_semantic_assurance
from project.solver.SIMP_MASTER import main_from_spec

OPT_OUT_PHRASES = {"use defaults", "use default", "skip", "just use defaults"}
CONFIRM_PHRASES = {"y", "yes", "confirm", "run"}
_PATH_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def _tokenize_path(path: str):
    tokens = []
    for name, index in _PATH_TOKEN.findall(path):
        tokens.append(int(index) if index else name)
    if not tokens:
        raise ValueError(f"Invalid field path: {path}")
    return tokens


def _get_path(data, path: str):
    node = data
    for token in _tokenize_path(path):
        node = node[token]
    return node


def _set_path(data, path: str, value):
    tokens = _tokenize_path(path)
    node = data
    for token in tokens[:-1]:
        node = node[token]
    node[tokens[-1]] = value


def _coerce_answer(raw: str, default):
    if isinstance(default, bool):
        value = raw.strip().lower()
        if value in {"true", "yes", "1"}:
            return True
        if value in {"false", "no", "0"}:
            return False
        raise ValueError("expected yes/no or true/false")
    if isinstance(default, int) and not isinstance(default, bool):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, str):
        return raw
    return json.loads(raw)


def _same_value(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1.0e-12, abs_tol=1.0e-12)
    return a == b


def _apply_overrides(spec, overrides: dict):
    payload = spec.model_dump()
    for path, value in overrides.items():
        _set_path(payload, path, value)
    return type(spec).model_validate(payload)


def main():
    prompt = input("Describe your topology optimization problem:\n> ")
    spec, defaulted_fields, field_provenance, parser_usage = parse_problem(prompt)

    clarification_policy = "ask_all_with_opt_out"
    clarifications_presented = []
    confirmed_defaults = []
    accepted_remaining_defaults = []
    user_overrides = []
    invalid_responses = []
    accepted_after_invalid_input = []
    opted_out = False
    opted_out_at_field = None
    overrides = {}
    spec_payload = spec.model_dump()

    if defaulted_fields:
        print(
            f"\n{len(defaulted_fields)} field(s) were defaulted. Press Enter "
            "to confirm one, enter a replacement value, or type 'use defaults' "
            "to accept all remaining defaults.\n"
        )

        for index, field in enumerate(defaulted_fields):
            if opted_out:
                accepted_remaining_defaults.append(field.field_path)
                print(f"  Using default for {field.field_path}: {field.default_used}")
                continue

            clarifications_presented.append(field.model_dump())
            answer = input(f"  {field.question}\n  > ").strip()
            if answer.lower() in OPT_OUT_PHRASES:
                opted_out = True
                opted_out_at_field = field.field_path
                accepted_remaining_defaults.extend(
                    remaining.field_path for remaining in defaulted_fields[index:]
                )
                print("  Using defaults for all remaining fields.\n")
                break

            actual_default = _get_path(spec_payload, field.field_path)
            if answer == "":
                confirmed_defaults.append(field.field_path)
                continue

            try:
                new_value = _coerce_answer(answer, actual_default)
                if _same_value(new_value, actual_default):
                    confirmed_defaults.append(field.field_path)
                    continue
                overrides[field.field_path] = new_value
                user_overrides.append(
                    {
                        "field_path": field.field_path,
                        "previous_value": actual_default,
                        "new_value": new_value,
                    }
                )
            except (
                ValueError,
                TypeError,
                KeyError,
                IndexError,
                json.JSONDecodeError,
            ) as exc:
                print(
                    f"  Could not apply '{answer}' to {field.field_path} ({exc}); "
                    f"keeping default {field.default_used}."
                )
                invalid_responses.append(
                    {
                        "field_path": field.field_path,
                        "response": answer,
                        "error": str(exc),
                    }
                )
                accepted_after_invalid_input.append(field.field_path)

    if overrides:
        spec = _apply_overrides(spec, overrides)

    inferred = [
        item
        for item in field_provenance
        if item.source in {
            "inferred_from_benchmark_name",
            "inferred_from_language",
        }
    ]
    if inferred:
        print("\nFields inferred rather than stated explicitly:")
        for item in inferred:
            print(
                f"  {item.field_path} = {item.value!r} "
                f"[{item.source}; evidence: {item.evidence}]"
            )

    print("\nFinal parsed specification:")
    print(spec.model_dump_json(indent=2))
    confirmation = input("\nRun this exact specification? [y/N]\n> ").strip().lower()
    final_preview_confirmed = confirmation in CONFIRM_PHRASES
    if not final_preview_confirmed:
        print("Run cancelled before deterministic optimization.")
        return

    semantic_assurance = summarize_semantic_assurance(
        field_provenance,
        final_preview_confirmed=final_preview_confirmed,
    )
    final_payload = spec.model_dump()
    override_paths = {item["field_path"] for item in user_overrides}
    final_field_provenance = []
    for item in field_provenance:
        record = item.model_dump()
        record["final_value"] = _get_path(final_payload, item.field_path)
        if item.field_path in override_paths:
            record["interaction_status"] = "user_overridden"
        elif item.field_path in confirmed_defaults:
            record["interaction_status"] = "individually_confirmed_default"
        elif item.field_path in accepted_remaining_defaults:
            record["interaction_status"] = "accepted_after_opt_out"
        elif item.field_path in accepted_after_invalid_input:
            record["interaction_status"] = "default_retained_after_invalid_input"
        elif item.source in {
            "inferred_from_benchmark_name",
            "inferred_from_language",
        }:
            record["interaction_status"] = "confirmed_in_final_preview"
        else:
            record["interaction_status"] = "unchanged"
        final_field_provenance.append(record)

    provenance = {
        "clarification_policy": clarification_policy,
        "defaulted_fields": [field.model_dump() for field in defaulted_fields],
        "parser_field_provenance": [item.model_dump() for item in field_provenance],
        "final_field_provenance": final_field_provenance,
        "clarifications_presented": clarifications_presented,
        "confirmed_defaults": confirmed_defaults,
        "accepted_remaining_defaults": accepted_remaining_defaults,
        "user_overrides": user_overrides,
        "invalid_responses": invalid_responses,
        "accepted_after_invalid_input": accepted_after_invalid_input,
        "opted_out": opted_out,
        "opted_out_at_field": opted_out_at_field,
        "final_preview_confirmed": final_preview_confirmed,
        "confirmation_received": final_preview_confirmed,
        "semantic_assurance": semantic_assurance,
        "original_prompt": prompt,
    }

    print("\nStarting SIMP optimization...\n")
    main_from_spec(spec, parser_usage=parser_usage, run_provenance=provenance)


if __name__ == "__main__":
    main()
