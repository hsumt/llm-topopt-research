"""Manual parser smoke tests. These call the live Anthropic API."""

from client import parse_problem

EXAMPLES = [
    "Cantilever beam, fixed left edge, 1 N downward point force at the midpoint of the right edge, mesh 60x20, volume fraction 0.5",
    "MBB beam, half symmetry, point load downward at top-left, mesh 120x40, volume fraction 0.4",
]

for prompt in EXAMPLES:
    spec, defaulted_fields, field_provenance, usage = parse_problem(prompt)
    print(spec.model_dump_json(indent=2))
    print("defaulted:", [f.field_path for f in defaulted_fields])
    print("provenance:")
    for item in field_provenance:
        print(f"  {item.field_path}: {item.source} <- {item.evidence}")
    print("tokens:", usage)
