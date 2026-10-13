# llm-topopt-research — Agentic Topology Optimization (ATO)

Agentic Topology Optimization puts language-model agents alongside an engineer across
the whole topology-optimization workflow. That runs from forming the problem
specification, to configuring and running the solver that implements it, to
diagnosing the result afterwards.

## How it works

The work is divided like this:
- **The agent makes the decisions.** It interprets the brief, fills the specification,
  and chooses how the codebase should realize it.
- **The codebase provides the means.** It offers modular, configurable building blocks
  for geometry, loads, supports, materials, objectives, constraints and discretization.
  It validates every request and runs it deterministically.
- **Results come only from the solver.**
- **A human keeps authority** over what is solved.

ATO has two halves:

1. **Pre-solve formulation.** An engineer describes a design problem in plain
   language. The system:
   - turns it into an explicit specification;
   - asks for the missing engineering decisions, without inventing physical values;
   - configures the solver from the specification;
   - shows the exact solver inputs for approval before anything runs.
2. **Post-solve specification diagnosis.** A run can converge and still be wrong,
   because the problem it solved was wrong. ATO:
   - analyses the numerical results and the data recorded during the run;
   - proposes edits to the problem;
   - accepts an edit only once a re-solve confirms it;
   - flags any edit that would change the engineer's intent for a human to decide.

The long-term aim is to run any topology-optimization problem an engineer can
describe, not one benchmark.

## Status

The repository is being cleaned up and restructured. Detailed usage documentation will
return as that work lands.

## Quick start

Docker (Engine with Compose, or Docker Desktop) is the only host dependency. It runs
on Linux, on Windows through WSL2, and on macOS; Apple Silicon Macs run it under x86
emulation, which is slower.

```bash
./docker/compose up --build -d
```

Open http://localhost:8501 (set `ATO_PORT` to use another port). The model features
need `ANTHROPIC_API_KEY`, either exported in your shell or set in the gitignored
`.env` file at the repository root. Never commit it.

```bash
./docker/compose down
```

## License

MIT (see `LICENSE`). The vendored ParaLeSTO sources under `_external/pyparalesto` are
Apache-2.0.
