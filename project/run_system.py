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
    Cantilever beam benchmark (Sigmund 2001 style): rectangular domain 1.6×1.0, mesh 80×50 elements, fully clamped left edge (u_x = u_y = 0), downward point load at the center of the right edge (x=1.6, y=0.5), volume fraction 0.4, SIMP penalty 3.0, filter radius 0.05, linear elastic material with E=1.0 and ν=0.3, max 200 optimization iterations, convergence tolerance 0.01.
    MBB beam benchmark (Sigmund 2001 style), half-symmetry model: rectangular domain 3.0×1.0, mesh 120×40 elements, symmetry plane on left edge (u_x = 0, u_y free), roller at bottom-right corner (u_y = 0, u_x free), downward point load at top-left corner (x=0, y=1.0), volume fraction 0.4, SIMP penalty 3.0, filter radius 0.06, linear elastic material with E=1.0 and ν=0.3, max 400 optimization iterations, convergence tolerance 0.01.
    Michell truss benchmark: rectangular domain 2.0×1.0, mesh 120×60 elements, simply supported at both bottom corners (pinned at bottom-left, roller at bottom-right), downward point load at the top-center node, volume fraction 0.3, SIMP penalty 4.0, filter radius 1.2, linear elastic material with E=1.0 and ν=0.3, max 300 optimization iterations, convergence tolerance 0.001.
            """
    spec, parser_usage   = parse_problem(prompt)
    print("\nParsed spec:")
    print(spec.model_dump_json(indent=2))
    print("\nStarting SIMP optimization...\n")
    main_from_spec(spec, parser_usage=parser_usage)