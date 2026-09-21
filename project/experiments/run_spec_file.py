"""Run a hand-written ProblemSpec JSON through main_from_spec.

Lives in project/experiments/, which is not hashed, so editing this file
never invalidates the verification or mesh-refinement manifests.
"""
import argparse
import json

from dotenv import load_dotenv

from project.parser.schema import ProblemSpec
from project.topopt.controller import main_from_spec

EXPLICIT_PROVENANCE = {
    "clarification_policy": "hand_written_spec",
    "defaulted_fields": [],
    "parser_field_provenance": [],
    "final_field_provenance": [],
    "clarifications_presented": [],
    "confirmed_defaults": [],
    "accepted_remaining_defaults": [],
    "accepted_after_invalid_input": [],
    "user_overrides": [],
    "invalid_responses": [],
    "opted_out": False,
    "opted_out_at_field": None,
    "final_preview_confirmed": True,
    "confirmation_received": True,
    "semantic_assurance": {
        "status": "fully_explicit",
        "confirmation_required": False,
        "final_preview_confirmed": True,
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec_json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--assume-explicit", action="store_true")
    args = ap.parse_args()

    load_dotenv()
    with open(args.spec_json) as handle:
        spec = ProblemSpec.model_validate(json.load(handle))

    main_from_spec(
        spec,
        out_dir=args.out,
        run_provenance=EXPLICIT_PROVENANCE if args.assume_explicit else None,
    )


if __name__ == "__main__":
    main()
