# llm-topopt-research

Specification-adequacy diagnosis for structural topology optimization.

The premise: a topology optimization run can converge correctly, satisfy every stated
constraint, and still return a design that is wrong — because the *specification* was
inadequate, not the solver. No check that reads only the stated problem can detect that. This
repository builds a benchmark where that situation is reproducible, and a cascade that
diagnoses it.

Branch `experiment/L_bracket_ATO` uses the L-bracket of

> Holmberg E., Torstenfelt B., Klarbring A. (2013). Stress constrained topology optimization.
> *Structural and Multidisciplinary Optimization* 48:33–47.

whose re-entrant corner is the canonical case where a stiffness-optimal design is a correct
answer to the wrong question.

---

## Layout

| Path | Role |
|---|---|
| `project/topopt/stress/` | Stress-constrained solver. Q4 plane stress (Holmberg's own model) and a 2.5D extruded solid model, behind one evaluator |
| `project/ato/` | The diagnosis cascade: predicates, edit grammar, hypothesis generation, capability matching, tool synthesis, discharge, ranking, authority |
| `project/experiments/lbracket/` | Runnable experiments and the declared design-intent document |
| `project/verification/` | Pre-existing V&V suite, including the independently verified Q4 element reference the new solver reuses |
| `project/topopt/`, `project/parser/`, `project/llm/`, `project/apps/` | The original DOLFINx compliance pipeline and NL-parsing front end, unchanged |
| `Archive/` | Frozen history. Read-only |

## The solver

Holmberg's formulation, not compliance with a stress plot added afterwards: bilinear elements
with one stress evaluation point at the centroid, cone design-variable filter, SIMP stiffness
penalization `q = 3`, stress penalization `η_S = ρ^(1/2)`, clustered modified P-norm stress
measure, stress-level clustering with reclustering every iteration, adjoint sensitivities, and
formulations P1 (min mass s.t. stress), P2 (min compliance s.t. stress and mass) and P3 (min
compliance s.t. mass). numpy and scipy only; DOLFINx is not required.

Two analysis models share one evaluator, so results are comparable cell by cell:

- `plane_stress` — 2 dof/node, 3 stress components. Holmberg's model.
- `extruded_3d` — the same in-plane grid swept through the thickness into layers of trilinear
  hexes, density held constant through the thickness. 3 dof/node, 6 stress components, so an
  out-of-plane load case is expressible. Design variable count unchanged; in-plane response
  within 0.3% of plane stress (measured).

A load case the model cannot represent raises `ModelCannotRepresent` rather than returning a
misleading number. A plane-stress node has no `w`, so a transverse load has no degree of
freedom to act on: the model does not answer weakly, it cannot be asked.

## The cascade

`project/ato/open_cascade.py`:

1. **Convergence gate** — the stated problem must converge feasibly. A failed run belongs to
   consistency repair, not adequacy diagnosis.
2. **Anomaly detection** — deterministic predicates on the converged field, plus a
   specification-completeness check against declared intent.
3. **Blind hypothesis generation** — a model proposes what would have to be different about
   the stated problem, **without being shown the edit catalogue**. Showing it the catalogue
   caps the hypothesis space at what the toolbox already does.
4. **Capability matching** — does a tool for that hypothesis exist? A partial match counts as
   a gap.
5. **Tool synthesis** — where none exists, a model writes a new edit operation, admitted only
   after an AST whitelist and a smoke test. A synthesized operation may only mutate the
   specification: it cannot compute, fetch, or assert anything, so it can never influence its
   own verdict.
6. **Deductive discharge** — apply each candidate *and each subset*, re-solve, and keep it
   only if the anomaly is entailed away. Subsets matter: a candidate eliminated alone can
   re-enter in combination.
7. **Ranking and authority** — lexicographic by admissibility tier, then edit count.
   Intent-altering (T3) survivors are ruled on against the written intent document, never on
   engineering plausibility.
8. **Closure** — apply what the intent authorises, re-solve, and report whether every declared
   load axis is within the allowable.

Two invariants hold regardless of what any model returns:

- **Tier is not delegated.** Admissibility is assigned from the problem-tuple slot and the
  declared intent (`grammar.tier_for_slot`). A proposer's suggested tier is recorded and
  overridden — measured disagreement has been substantial, and a stage that set its own tier
  could route an intent-altering edit past the authority gate.
- **Verdicts come only from re-solving.** No model decides whether a hypothesis survives.

## Design intent is an input

`project/experiments/lbracket/intent_lbracket.json` states what the part must do, independent
of how the problem was written down: which load axes it carries, which geometry is fixed and
which is the designer's to choose, whether stated limits are firm. This is not a result and is
not inferred from any run — it is what makes "intent-revealing" and "intent-altering"
separable at all, and it is the only source of authority for adopting a T3 edit.

## Running

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # the solver is serial

python -m project.experiments.lbracket.run_ato --check-backend      # validate the API key path
python -m project.topopt.stress.verify_gradients                    # adjoint vs finite differences
python -m project.topopt.stress.verify_extrusion                    # 2.5D model vs plane stress and beam theory
python -m project.experiments.lbracket.run_benchmark --fidelity low # P1/P2/P3 comparison
python -m project.experiments.lbracket.run_paper_comparison         # against the published tables
python -m project.experiments.lbracket.run_open_ato --max-rounds 2  # the open cascade
```

Requires `anthropic` and `python-dotenv` in addition to numpy/scipy. The key is read from the
process environment; no code here reads `.env` directly.

## Status

Findings, limitations, and the quantitative comparison against the published tables — which
does **not** currently reproduce — are in `RESULTS_L_bracket_ATO.md`. Read it before extending
this work; several results are qualified and one is an open discrepancy.
