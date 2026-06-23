import sys

sys.path.insert(0, "project/parser")
sys.path.insert(0, "project/solver")


from client import parse_problem
from SIMP_MASTER import main_from_spec
# make it so its more dynamic for different writing styles and also runs a baseline if not enough entered / runs the optimization max-iterations like fr.
if __name__ == "__main__":
    prompt = input("Describe your topology optimization problem:\n> ")
    """
Example Tests:
#1: Success
        Cantilever beam benchmark (Sigmund 2001 style): rectangular domain 1.6×1.0, mesh 80×50 elements, fully clamped left edge (u_x = u_y = 0), downward point load at the center of the right edge (x=1.6, y=0.5), volume fraction 0.4, SIMP penalty 3.0, filter radius 0.05, linear elastic material with E=1.0 and ν=0.3, max 200 optimization iterations, convergence tolerance 0.01.
        
#2: Success
        MBB beam benchmark (Sigmund 2001 style), half-symmetry model: rectangular domain 3.0×1.0, mesh 120×40 elements, symmetry plane on left edge (u_x = 0, u_y free), roller at bottom-right corner (u_y = 0, u_x free), downward point load at top-left corner (x=0, y=1.0), volume fraction 0.4, SIMP penalty 3.0, filter radius 0.06, linear elastic material with E=1.0 and ν=0.3, max 400 optimization iterations, convergence tolerance 0.01.
        
#3: Fail
        Make me a cantilever beam. The left side should be fixed and there should be a downward force on the middle of the right side. Use about 40% material and try to make it lightweight but stiff.
        
#4: Fail
        I want a bridge-like structure. Support it on both ends and put a downward force in the middle. Use around 30% material.
        
#5: Success
        Design a 2D compliance-minimizing cantilever beam using the SIMP method.

        Domain dimensions shall be 1.6 units by 1.0 units discretized into an 80 by 50 quadrilateral mesh.

        Apply homogeneous Dirichlet boundary conditions on the entire left boundary so that ux = uy = 0.

        Apply a concentrated vertical load Fy = -1.0 at the geometric midpoint of the right edge.

        Material properties:
        E = 1.0
        nu = 0.3

        Optimization settings:
        volume fraction = 0.4
        penalization exponent = 3.0
        filter radius = 0.05
        maximum iterations = 200
        convergence tolerance = 0.01

#6:
        Optimize a cantilever beam.

#7:
        Optimize a cantilever beam with 40% material.
#8:
        Take a rectangular plate.

        Clamp the left side.

        Push down near the far end.

        Use about half the material.

        Make it as stiff as possible.
#9:
        A beam is rigidly attached on the left and loaded vertically downward at the free end.

        Find the best material layout using 40% of the domain.
#10:
        Optimize a rectangular beam. Use 40% material.

#11:
        Fix the left edge and also allow it to move freely. Apply a downward load and no load.

#12: 
        I am designing a lightweight bracket to hold my diaper changing station up, with the left side is bolted to a wall. A downward force acts near the right side. Use at most 40% material and make the structure as stiff as possible.


Cycled out:
        Michell truss benchmark: rectangular domain 2.0×1.0, mesh 120×60 elements, simply supported at both bottom corners (pinned at bottom-left, roller at bottom-right), downward point load at the top-center node, volume fraction 0.3, SIMP penalty 4.0, filter radius 1.2, linear elastic material with E=1.0 and ν=0.3, max 300 optimization iterations, convergence tolerance 0.001.
            """
    spec, parser_usage   = parse_problem(prompt)
    print("\nParsed spec:")
    print(spec.model_dump_json(indent=2))
    print("\nStarting SIMP optimization...\n")
    main_from_spec(spec, parser_usage=parser_usage)