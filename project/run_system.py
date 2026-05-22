import sys

sys.path.insert(0, "project/parser")
sys.path.insert(0, "project/solver")


from client import parse_problem
from SIMP_MASTER import main_from_spec

if __name__ == "__main__":
    prompt = input("Describe your topology optimization problem:\n> ")
    """
    For example:
    Cantilever beam, fixed left edge, 1N downward at tip, mesh 60x20, vol_frac 0.5
    """
    spec   = parse_problem(prompt)
    print("\nParsed spec:")
    print(spec.model_dump_json(indent=2))
    print("\nStarting SIMP optimization...\n")
    main_from_spec(spec)