# 3D L-bracket integration context

Reviewed 2026-10-02. This note records the decisions behind importing
`p_norm_stress_constraint/lbracket/simplified_3D_holes/` into the current
pre-solve application. It does not import conversation transcripts or treat
illustrative slide results as numerical evidence.

## Requested behavior

An engineer describes a 3D L-bracket in ordinary language. The application
extracts the stated specification, asks for missing engineering decisions,
retains the answers, shows the resolved inputs, and runs the supported script
after explicit approval. The solver receives a validated numeric input
contract with units. Readiness must check that contract independently of
whether an LLM reports any issues.

The 2026-10-02 request also calls for a clean Docker setup containing ParaLeSTO
under `_external`, so execution must not depend on a neighboring checkout or
an existing machine-specific container.

## Imported benchmark and its boundaries

The source is the neighboring MFRTO repository's
`p_norm_stress_constraint/lbracket/simplified_3D_holes/main.py`, with
`pyfea.py` and `utils.py`. The source identifies N. Jurado's L-bracket changes
to the thermo-mechanical level-set example attributed to A. Guibert,
M. Pozzi, and M. Bookwala. Preserve these attributions.

| Quantity | Source choice |
|---|---|
| Domain | 100 × 100 × 12 mm bounding box; 60 × 60 mm upper-right cut |
| Grid | 50 × 50 × 6 cubic cells |
| Material | E = 120 GPa, Poisson ratio = 0.36 |
| Support | All translations fixed on the top face of the vertical arm |
| Load | 5,000 N downward, distributed over a 4 mm band at the horizontal tip, across the full thickness |
| Initial holes | Five through-thickness cylinders, radius 10 mm |
| Hole centers, mm | (20,20), (20,50), (20,80), (50,20), (80,20) |
| Objective | Full compliance, Fᵀu |
| Constraints | Volume ≤ 75% of the L-domain volume; normalized p-norm stress ≤ 116 MPa, p = 6 |

The five cylinders are **initial design voids**. The optimizer may move,
merge, or close them. They are not mounting holes that must remain open.
The corner cut is a permanent void, and the load application cuboid is
protected solid. A request for fixed mounting holes is a different geometry
requirement and must not silently map to these initial holes.

The stress quantity is
`Sp = [integral((rho * von_Mises)^p) / V_L]^(1/p)`.
The 116 MPa limit comes from the initial stress calibration divided by 0.92;
it is neither a material yield strength nor a bound on every local stress.
The user chose the five-hole pattern from Kambampati, Chung & Kim (2021),
Fig. 5, with a five-element radius, and retained the parent's 75% volume
limit and p = 6. Dimensions read approximately from that figure were
ratified in the source conversation. This is a 3D extension, not an
established reproduction of the paper.

This backend is separate from the older Holmberg Q4 P3 benchmark on
`experiment/L_bracket_ATO`. That cascade's P3 baseline minimizes
half-compliance under a mass constraint and deliberately omits stress
constraints. Its predicates, convergence evidence, candidate verdicts,
and authority decisions cannot be transferred to this 3D level-set run.
The source 3D script uses an iteration cap, not a convergence stopping test.
Completing a bounded run does not establish convergence or design validity.

## Requirements recovered from the presentations

- `Monthly_Meeting_01_Quantum_Computing_&_Agentic_AI.pptx`, slides 4–5:
  pre-run clarification precedes human approval and TO/FEA; the intended
  interface uses a solver-independent `ProblemSpec`, source provenance,
  targeted updates with revalidation, and deterministic readiness.
- `Pre-Solve_ATO_Meeting_filled.pptx`, slides 2 and 8:
  the parser records stated decisions; the critic asks about unresolved
  physical intent, with at most four questions per round; the resolver
  proposes updates and Python applies them. Solver settings should not
  become unnecessary engineering questions.
- `Pre-Solve_ATO_Meeting_draft2.pptx`, slides 4–6 and 7:
  the reviewed application lost clarification context, could re-ask
  answered questions, checked too few fields for readiness, had no numeric
  solver adapter, and retained sessions only in Streamlit memory.
  These are pre-integration findings, not a claim about the completed port.
- `Pre-Solve_ATO_Meeting_filled.pptx`, slides 12–15:
  the disabled solver action needs a real handoff; history should retain
  request, reviews, answers, changes, and an approval tied to the exact
  specification. Slide 15 names grid, node sets, and load vectors as adapter
  outputs.
- `Pre-Solve_ATO_Meeting_draft2.pptx`, slide 10 and its notes:
  the solver contract must account for domain/mesh, supports, loads,
  volume or mass fraction, objective, and solver controls with units.

The August worked-example deck explicitly labels its discharge outcomes
as predictions (slides 1 and 24). Its L-bracket question about transverse
load direction, location, magnitude, and corner fillet (slide 14) explains
the intended authority boundary. It does not authorize adding an unstated
load or hardcoding a candidate's outcome. The August proposal deferred
full 3D; the current request explicitly advances that scope.

## Local evidence index

These sources are outside the repository and are listed for traceability;
they are not required to build or run it.

- Source repository: sibling project
  `01_Multifidelity_Robust_Topology_Optimization/mfrto/`.
  Source `main.py` lines 74–109 specify geometry/material, 150–200 the
  supports/load, 215–228 the objective/stress measure, 254–260 void/solid
  behavior, and 283–292 constraints and iteration cap.
- Claude source session, 2026-10-02:
  `~/.claude/projects/-home-nelsonj-Documents-01-M2DO-01-Research-01-Projects-01-Multifidelity-Robust-Topology-Optimization-mfrto/e6d8aa86-f7b8-488d-8379-730b5237d2c0.jsonl`.
  The bracket decisions appear around lines 3209–3522.
- Claude project session, 2026-10-02:
  `~/.claude/projects/-home-nelsonj-Documents-01-M2DO-01-Research-01-Projects-02-Agentic-Topology-Optimization-llm-topopt-research/e8c0d94c-1fe9-4374-acf7-c627bbf7d21c.jsonl`.
  The stated aim includes script-by-script interrogation and visualization.
- Meeting decks: `~/Downloads/`, exact filenames listed above.
  Both Pre-Solve decks were saved on 2026-10-02 and reviewed the
  pre-integration branch at `ea35d9f`.
- Proposal deck: `~/Downloads/ATO_Proposal_Slide_Deck (3).pptx`,
  saved 2026-08-31; slides 7–10 describe the proposed cascade and scope.
- Worked examples: `~/.claude-science/orgs/24da4a97-ca1b-475b-8b8e-5abdb105d186/artifacts/proj_3500ab8d19e5/c3001ea0-eb37-4efa-8710-d3b45f25de4e/v7d8e1534_ATO_worked_examples_deck.pptx`,
  saved 2026-08-23. Embedded L-bracket diagrams were inspected as well as
  text and notes. Slide numbers here follow presentation order.
