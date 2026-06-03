import sys

sys.path.insert(0, "project/parser")
sys.path.insert(0, "project/solver")


from client import parse_problem
from SIMP_MASTER import main_from_spec
# make it so its more dynamic for different writing styles and also runs a baseline if not enough entered / runs the optimization max-iterations like fr.
if __name__ == "__main__":
    prompt = input("Describe your topology optimization problem:\n> ")
    """
    For example:
    Cantilever beam, fixed left edge, 1N downward at tip, mesh 60x20, vol_frac 0.5
    Cantilever beam benchmark (Sigmund 2001 style): rectangular domain 1.6×1.0, mesh 80×50 elements, fixed left edge, downward point load at the right tip, volume fraction 0.4, SIMP penalty 3.0, filter radius 1.5, linear elastic material with E=1.0 and ν=0.3, max 100 optimization iterations, convergence tolerance 0.01.
    MBB beam benchmark (Sigmund 2001 style): rectangular domain 6.0×1.0, mesh 180×60 elements, simply supported at both bottom corners (pinned at bottom-left, roller at bottom-right), downward point load at the top-center node, volume fraction 0.5, SIMP penalty 3.0, filter radius 2.0, linear elastic material with E=1.0 and ν=0.3, max 600 optimization iterations, convergence tolerance 0.001.
    Michell truss benchmark: rectangular domain 2.0×1.0, mesh 120×60 elements, simply supported at both bottom corners (pinned at bottom-left, roller at bottom-right), downward point load at the top-center node, volume fraction 0.3, SIMP penalty 4.0, filter radius 1.2, linear elastic material with E=1.0 and ν=0.3, max 300 optimization iterations, convergence tolerance 0.001.
            """
    spec   = parse_problem(prompt)
    print("\nParsed spec:")
    print(spec.model_dump_json(indent=2))
    print("\nStarting SIMP optimization...\n")
    main_from_spec(spec)