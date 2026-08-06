"""Manual parser smoke tests that call the live Anthropic API.

Run from the repository root:

    /dolfinx-env/bin/python -m project.experiments.parser_live_smoke
"""

from project.parser.client import parse_problem


EXAMPLES = [
    (
        "Cantilever beam, fixed left edge, 1 N downward point force at "
        "the midpoint of the right edge, mesh 60x20, volume fraction 0.5"
    ),
    (
        "MBB beam, half symmetry, point load downward at top-left, "
        "mesh 120x40, volume fraction 0.4"
    ),
]


def main() -> None:
    for prompt in EXAMPLES:
        spec, defaulted_fields, field_provenance, usage = parse_problem(prompt)

        print(spec.model_dump_json(indent=2))
        print("defaulted:", [field.field_path for field in defaulted_fields])
        print("provenance:")

        for item in field_provenance:
            print(
                f"  {item.field_path}: "
                f"{item.source} <- {item.evidence}"
            )

        print("tokens:", usage)


if __name__ == "__main__":
    main()