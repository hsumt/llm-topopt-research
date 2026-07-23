"""Interactive natural-language runner for the verified 2-D SIMP pipeline.
cd /workspaces/llm-topopt-research
/dolfinx-env/bin/python project/solver/verification_tests.py
cd /workspaces/llm-topopt-research
/dolfinx-env/bin/python project/run_system.py
"""

from __future__ import annotations

import json
import re
import sys

sys.path.insert(0, "project/parser")
sys.path.insert(0, "project/solver")

from client import parse_problem
from SIMP_MASTER import main_from_spec

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
    """Coerce an override to the type of the actual defaulted value."""
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
    # Fallback permits JSON arrays/objects if the schema later adds them.
    return json.loads(raw)


def _apply_overrides(spec, overrides: dict):
    payload = spec.model_dump()
    for path, value in overrides.items():
        _set_path(payload, path, value)
    return type(spec).model_validate(payload)


def main():
    prompt = input("Describe your topology optimization problem:\n> ")
    spec, defaulted_fields, parser_usage = parse_problem(prompt)

    clarification_policy = "ask_all"
    clarifications_presented = []
    user_overrides = []
    accepted_defaults = []

    if defaulted_fields:
        print(
            f"\n{len(defaulted_fields)} field(s) were defaulted. Press Enter "
            "to accept one, enter a replacement value, or type 'use defaults' "
            "to accept all remaining defaults.\n"
        )
        opted_out = False
        overrides = {}
        spec_payload = spec.model_dump()

        for field in defaulted_fields:
            if opted_out:
                print(f"  Using default for {field.field_path}: {field.default_used}")
                continue

            clarifications_presented.append(field.model_dump())
            answer = input(f"  {field.question}\n  > ").strip()
            if answer.lower() in OPT_OUT_PHRASES:
                opted_out = True
                print("  Using defaults for all remaining fields.\n")
                continue
            if answer == "":
                accepted_defaults.append(field.field_path)
                continue

            try:
                actual_default = _get_path(spec_payload, field.field_path)
                new_value = _coerce_answer(answer, actual_default)
                overrides[field.field_path] = new_value
                user_overrides.append({
                    "field_path": field.field_path,
                    "previous_value": actual_default,
                    "new_value": new_value,
                })
            except (ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError) as exc:
                print(
                    f"  Could not apply '{answer}' to {field.field_path} ({exc}); "
                    f"keeping default {field.default_used}."
                )

        if overrides:
            spec = _apply_overrides(spec, overrides)

    accepted_defaults = [
        field.field_path for field in defaulted_fields
        if field.field_path not in {item["field_path"] for item in user_overrides}
    ]

    print("\nParsed spec:")
    print(spec.model_dump_json(indent=2))
    print("\nStarting SIMP optimization...\n")
    provenance = {
        "clarification_policy": clarification_policy,
        "defaulted_fields": [field.model_dump() for field in defaulted_fields],
        "clarifications_presented": clarifications_presented,
        "accepted_defaults": accepted_defaults,
        "user_overrides": user_overrides,
        "confirmation_received": False,
        "original_prompt": prompt,
    }
    main_from_spec(spec, parser_usage=parser_usage, run_provenance=provenance)


if __name__ == "__main__":
    main()
