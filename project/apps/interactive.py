"""Interactive natural-language runner for the verified 2-D SIMP pipeline.

Pipeline:
    natural language
      -> parser + default clarification
      -> interactive formulation critic/resolver loop
      -> deterministic intent preview + human confirmation
      -> deterministic FEniCS/SIMP

Run from the repository root:

    /dolfinx-env/bin/python -m project.apps.interactive
"""

from __future__ import annotations

import json
import math
import re

from project.formulation.dialogue import run_formulation_dialogue
from project.formulation.patching import build_review_field_provenance
from project.parser.client import parse_problem

OPT_OUT_PHRASES = {"use defaults", "use default", "skip", "just use defaults"}
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

    clarification_policy = "parser_defaults_then_interactive_formulation_review"
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

    # Existing parser/default clarification stage. This remains distinct from
    # the formulation critic: parser defaults are field-value gaps, while the
    # formulation critic asks whether the resulting engineering idealization is
    # actually what the user intended.
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
                        "response": answer,
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
        print("\nFields initially inferred rather than stated explicitly:")
        for item in inferred:
            print(
                f"  {item.field_path} = {item.value!r} "
                f"[{item.source}; evidence: {item.evidence}]"
            )

    print("\nInitial parsed specification:")
    print(spec.model_dump_json(indent=2))

    review_field_provenance = build_review_field_provenance(
        spec=spec,
        parser_field_provenance=field_provenance,
        confirmed_defaults=confirmed_defaults,
        accepted_remaining_defaults=accepted_remaining_defaults,
        accepted_after_invalid_input=accepted_after_invalid_input,
        user_overrides=user_overrides,
    )

    formulation = run_formulation_dialogue(
        original_prompt=prompt,
        spec=spec,
        final_field_provenance=review_field_provenance,
    )
    if formulation.get("cancelled"):
        print(
            "Formulation session stopped before deterministic optimization. "
            f"Session artifacts: {formulation.get('session_dir')}"
        )
        return

    spec = formulation["spec"]
    final_field_provenance = formulation["final_field_provenance"]
    semantic_assurance = formulation["semantic_assurance"]

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
        "final_preview_confirmed": True,
        "confirmation_received": True,
        "semantic_assurance": semantic_assurance,
        "formulation_critique": {
            "policy": "interactive_critic_resolver_human_gate",
            "reviewed_before_run": True,
            "human_acknowledged": True,
            "result": formulation["last_critique"],
            "usage": formulation["usage"],
            "critic_calls": formulation["critic_calls"],
            "resolution_audits": formulation["resolution_audits"],
            "accepted_as_is": formulation["accepted_as_is"],
        },
        "formulation_session": {
            "session_id": formulation["session_id"],
            "session_dir": formulation["session_dir"],
            "intent_preview_path": formulation["preview_path"],
        },
        "original_prompt": prompt,
    }

    print("\nStarting deterministic SIMP optimization...\n")
    main_from_spec(spec, parser_usage=parser_usage, run_provenance=provenance)


if __name__ == "__main__":
    main()
