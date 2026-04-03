from client import parse_problem
from simp import SIMPSolver

def run_pipeline(user_input: str):
    print(f"--- Parsing User Intent ---")
    # 1. LLM parses the text into a structured object
    problem_spec = parse_problem(user_input)
    print(f"Problem parsed: {problem_spec.name}")

    # 2. Initialize the solver with the LLM's understanding
    solver = SIMPSolver(spec=problem_spec)

    # 3. Run the optimization (which includes your Agents)
    solver.optimize()

if __name__ == "__main__":
    user_prompt = "Cantilever beam, fixed left edge, 1N downward at tip, mesh 80x30, vol_frac 0.35"
    run_pipeline(user_prompt)