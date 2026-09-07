"""Live development evaluation for parser + pre-solve formulation critic.

This is a smoke/evaluation harness, not a publication benchmark. It calls the
live Anthropic API but does NOT run FEA/SIMP.

Run:
    /dolfinx-env/bin/python -m project.experiments.formulation_eval
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from project.formulation.patching import build_review_field_provenance
from project.llm.formulation_critic import critique_formulation
from project.parser.client import parse_problem
from project.paths import FORMULATION_EVALUATION_ROOT, PROJECT_ROOT

CASES_PATH = PROJECT_ROOT / "experiments" / "formulation_cases.txt"


def _load_cases() -> list[dict]:
    cases = []
    with CASES_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def main() -> None:
    records = []
    passed = 0

    for case in _load_cases():
        print(f"\n[{case['case_id']}] {case['prompt']}")
        record = {"case": case, "passed": False}
        try:
            spec, defaults, provenance, parser_usage = parse_problem(case["prompt"])
            record["parser_usage"] = parser_usage
            record["parse_rejected"] = False

            if case.get("expected_parse_reject"):
                record["failure"] = "expected parser to reject but it produced a spec"
                records.append(record)
                print("  FAIL expected parser rejection")
                continue

            final_provenance = build_review_field_provenance(
                spec=spec,
                parser_field_provenance=provenance,
                confirmed_defaults=[],
                accepted_remaining_defaults=[f.field_path for f in defaults],
                accepted_after_invalid_input=[],
                user_overrides=[],
            )
            critique, usage = critique_formulation(
                original_prompt=case["prompt"],
                spec=spec,
                final_field_provenance=final_provenance,
            )
            record["spec"] = spec.model_dump()
            record["critique"] = critique
            record["critic_usage"] = usage

            actual = {issue["category"] for issue in critique.get("issues", [])}
            required = set(case.get("required_categories", []))
            missing = sorted(required - actual)
            too_many = len(critique.get("issues", [])) > int(
                case.get("max_user_facing_issues", 4)
            )
            record["missing_required_categories"] = missing
            record["too_many_issues"] = too_many
            record["passed"] = not missing and not too_many

        except Exception as exc:
            record["parse_rejected"] = True
            record["exception"] = str(exc)
            record["passed"] = bool(case.get("expected_parse_reject"))

        if record["passed"]:
            passed += 1
            print("  PASS")
        else:
            print("  FAIL")
        records.append(record)

    total = len(records)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cases": total,
        "passed": passed,
        "failed": total - passed,
        "records": records,
    }
    FORMULATION_EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = FORMULATION_EVALUATION_ROOT / f"formulation_eval_{stamp}.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\nDevelopment evaluation: {passed}/{total} passed")
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
