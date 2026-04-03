# test_parser.py
from client import parse_problem

examples = [
    "Cantilever beam, fixed left edge, 1N downward at tip, mesh 60x20, vol_frac 0.5",
    "MBB beam, simply supported corners, uniform load on top, mesh 120x40, vol_frac 0.4",
    "Short column, fixed bottom, compressive load 5N top, mesh 20x20x20, vol_frac 0.3"
]

for prompt in examples:
    problem = parse_problem(prompt)
    print(problem.model_dump_json(indent=2))