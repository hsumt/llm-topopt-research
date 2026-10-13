# llm-topopt-research — Agentic Topology Optimization (ATO)

## What ATO is

Agentic Topology Optimization puts language-model agents alongside an engineer across
the whole topology-optimization workflow. That runs from forming the problem
specification, to configuring and running the solver that implements it, to
diagnosing the result afterwards.

This file describes the design ATO is being built toward. The repository implements
part of it; check the code before assuming a capability exists.

The division of labour:
- **The agent makes the decisions.** It interprets the engineer's brief, fills the
  specification, and chooses how the codebase should realize it. For example, it may
  decide to add a load case, a support, a material region, or a constraint.
- **The codebase provides the means.** It offers modular, configurable building blocks
  for each part of a problem: geometry, loads, supports, materials, objectives,
  constraints and discretization. It validates every request and runs it
  deterministically.
- **Results come only from the solver.** The agent never asserts a numerical outcome.
  It reasons from the data the solver records.
- **A human keeps authority** over what is solved.

**Design consequence.** Every element a specification can contain must map to a
solver-side capability the agent can invoke. When a specification needs something no
capability expresses, the system reports it as unsupported. That gap is then work for
the codebase. It is never silently dropped or approximated.

ATO has two halves:

1. **Pre-solve formulation.** An engineer describes a design problem in plain
   language. The system:
   - turns it into an explicit, machine-readable specification;
   - extracts only what was stated and never invents physical values;
   - asks for the engineering decisions that are missing;
   - configures the solver from the specification;
   - shows the exact solver inputs for human approval before anything runs.

   The model decides how the specification is realized; code checks that it can be;
   a person authorizes the run.
2. **Post-solve specification diagnosis.** A run can converge and be feasible yet
   still be wrong, because the problem it solved was wrong. For example, a
   minimum-compliance design can put material at a re-entrant corner and create a
   stress singularity. ATO looks for that kind of specification defect after
   convergence:
   - **Analyse the run data.** It reads the numerical results and data recorded during
     the run:
     - final fields, such as density, displacement and stress;
     - the histories of the objective, the constraints and the design change;
     - which constraints are active, and where.

     From these it looks for evidence that the specification, not the solve, is at
     fault. Each finding cites the data it rests on.
   - **Propose edits.** From that evidence, it proposes edits to the problem.
   - **Confirm by re-solve.** An edit counts only once a re-solve shows that the
     defect is gone from the new run's data.
   - **Escalate intent changes.** Edits that would change the engineer's intent are
     flagged for a human, not applied.

   This makes the run's outputs part of the interface. A solver must record enough
   data for the analysis to work from.

The long-term aim is to run any topology-optimization problem an engineer can
describe, not one benchmark.

## Hard rules

- Never read, print, or parse `.env` or any `*.env` file. The API key reaches the app
  only as an environment variable.
- Never write a credential into code, logs, artifacts, or commits.
- Do not edit `_external/pyparalesto/`. It is an unmodified vendored copy of upstream
  ParaLeSTO (Apache-2.0); keep it byte-identical.
