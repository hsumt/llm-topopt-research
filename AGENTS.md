# AGENTS.md — `llm-topopt-research`

Branch: `experiment/L_bracket_ATO`
Last refreshed: 2026-09-01. Supersedes any earlier local context file in this repo.

---

## 0. Hard rules

1. **Never read, open, cat, grep, print, parse, or otherwise ingest `.env` or any `*.env`
   file.** Not to check whether a key exists, not to debug an auth failure, not via a
   directory listing that dumps contents. Code obtains `ANTHROPIC_API_KEY` from the process
   environment at runtime (`dotenv.load_dotenv()` then `os.environ`); you never look at the
   value. `.env` is gitignored and stays that way.
2. **Never print, log, echo, or write a credential** into a file, artifact, notebook output,
   or commit.
3. `Archive/` is frozen history. Read for reference; do not edit, delete, or resurrect.
4. Before modifying anything listed in
   `project/verification/manifest.py::VERIFIED_RELATIVE_FILES`, read §6. Editing one
   invalidates the hash-bound verification manifest.
5. This file is gitignored — local context only. Anything that must travel with the branch
   goes in `README.md` or a committed note.

---

## 1. What this branch is

A **stress-constrained L-bracket benchmark aligned to the foundational paper**, and an
**agentic specification-adequacy cascade running on top of it**.

> Holmberg E., Torstenfelt B., Klarbring A. (2013). "Stress constrained topology
> optimization." *Struct Multidisc Optim* 48:33–47.

The paper is the right anchor because its headline comparison *is* the ATO premise. Its
formulation (P3) — minimum compliance subject to a mass limit, the traditional stiffness
statement — converges correctly and **places material in the re-entrant corner, producing a
geometric stress singularity**. Nothing about the solve is wrong. The design is optimal for
the problem as posed. The defect is in the specification, and no check that reads only the
stated problem can surface it. That is a converged, feasible run whose *specification* is at
fault, which is exactly what the cascade is built to diagnose.

### Deliberate design choice: the baseline is P3, not P1

The cascade is asked to diagnose **P3**. It is not told that stress constraints exist. Whether
adding them is part of the answer is something the discharge stage decides by re-solving. Do
not "help" the cascade by seeding P1 into the baseline.

---

## 2. Repository map

| Path | Role |
|---|---|
| `project/topopt/stress/` | **Holmberg formulation.** Q4 FEA, cone filter, penalized von Mises, clustering, adjoint, MMA loop |
| `project/ato/` | **The cascade.** Anomaly predicates, edit grammar, LLM hypothesis generation, discharge, ranking |
| `project/experiments/lbracket/` | Runners: `run_benchmark.py`, `run_ato.py` |
| `project/topopt/` | Original DOLFINx compliance path (`controller.py` authoritative), unchanged |
| `project/verification/` | Hard validation gate and numerical V&V; `reference_q4.py` is shared with the new solver |
| `project/parser/`, `project/llm/`, `project/apps/` | NL parsing, evidence-limited critic, entry points |
| `Archive/` | Frozen: old README, `.devcontainer/`, prior artifacts |

### `project/topopt/stress/` — module by module

| Module | Paper section | Contents |
|---|---|---|
| `domain.py` | Fig. 4 | L-shaped domain, structured Q4 mesh, clamp, load point, non-design patch, optional corner fillet |
| `fem_q4.py` | Sec. 1 | Assembly and solve; element stiffness reused from `verification/reference_q4.py` |
| `filters.py` | Sec. 3, Eq. (2) | Cone design-variable filter `rho = W x` |
| `stress.py` | Sec. 4.2, Sec. 5 | `eta_S = rho^(1/2)`, penalized von Mises and its derivative |
| `clustering.py` | Sec. 6 | `stress_level` and `distributed_stress` techniques |
| `mma_multi.py` | Sec. 7 | Multi-constraint driver over the vendored Svanberg `mmasub` |
| `problem.py` | Sec. 2 | `StressSpec` — the machine-readable problem tuple; fidelity tiers |
| `solver.py` | Sec. 2, 7 | `Evaluator` (objective/constraints/adjoint) and the MMA loop |
| `verify_gradients.py` | Sec. 7 | Finite-difference verification of the adjoint |
| `plotting.py` | — | Field rendering back onto the bounding grid |

---

## 3. The benchmark, as implemented

Values are Holmberg's unless marked. **Do not silently change any of them.**

| Quantity | Value | Source |
|---|---|---|
| Geometry | L-shape, overall `L × L`, both arms `2L/5` | Fig. 4 |
| `L` | 200 mm, thickness 1 mm | Sec. 9.1 |
| Material | E = 71 000 MPa, ν = 0.33, ρ = 2.8e-9 ton/mm³ | Sec. 9.1 (aircraft aluminium) |
| Yield / stress limit | 350 MPa | Sec. 9.1 |
| Load | 1500 N downward at the horizontal-arm tip | Fig. 4, Sec. 9.1 |
| Non-design region | 3 × 2 elements under the load | Sec. 8.1, Fig. 3 |
| Filter radius | `r0 = 1.5 ×` element size | Sec. 9.1 |
| SIMP stiffness penalty | `q = 3` | Sec. 4.1 |
| Stress penalty | `eta_S = rho^(1/2)` | Sec. 4.2, Eq. (4) |
| P-norm factor | `p = 8` | Sec. 6 |
| Clusters | `nc = 10`, stress-level, recluster every iteration | Sec. 6, Sec. 10 |
| Initial design | `rho = 0.5` | Sec. 9 |
| Formulations | P1 min mass s.t. stress; P2 min compliance s.t. stress + mass; P3 min compliance s.t. mass | Sec. 2 |

**Mesh check that must keep passing.** At `n_cells_per_side = 100` (h = 2 mm) the L-shaped
domain contains exactly **6400 elements**, which is the paper's stated discretisation. This is
a sharp check — only the 0.4/0.6 arm split reproduces it — so treat a change in that number as
a geometry regression.

### Fidelity tiers (`problem.py::FIDELITY`)

`arm_fraction × n_cells_per_side` must be an integer, or the re-entrant corner and the load
point do not land on node lines. For 0.4 that means **a multiple of 5**; `domain.py` raises
with that message rather than failing obscurely.

| Tier | grid | elements | role |
|---|---|---|---|
| `coarse` | 20 | 256 | smoke tests |
| `low` | 40 | 1024 | **the tier everything runs on** |
| `medium` | 60 | 2304 | sensitivity checks |
| `reference` | 100 | **6400** | Holmberg's own mesh — **bounded probe only** |

**The reference tier is a cost probe, not a production run.** This work has to run on a
laptop, and the cascade re-solves the problem once per candidate and once per subset. Use
`reference` with a small `max_iter` (`run_benchmark.reference_probe()` defaults to 1) to state
what a reference iteration costs on the machine at hand. Do not run a full reference
optimization inside the cascade.

---

## 4. The cascade

Five stages, in `project/ato/`. **The model's only role is stage 3.**

1. **Convergence gate** (`cascade.py`) — a baseline that did not converge on a feasible stated
   problem raises `ConvergenceGateRefusal`. Failed runs are consistency repair, not adequacy
   diagnosis.
2. **Anomaly detection** (`anomaly.py`) — deterministic predicates, see §5.
3. **Hypothesis generation** (`hypothesize.py`) — the model proposes candidate edits against
   the grammar. It is told explicitly that it does not decide survival and does not decide
   tier. Malformed candidates are fed back once for repair rather than silently dropped.
4. **Deductive discharge** (`discharge.py`) — every candidate and every subset up to
   `max_subset_size` is applied and **re-solved from scratch**. A subset survives if none of
   the targeted anomalies fire again. A re-solve that ends stress-infeasible yields
   `NOT_ASSESSED` with a reason — an absence of evidence, never an elimination.
5. **Ranking and authority gate** (`rank.py`) — lexicographic by tier then edit count. T3
   intent-altering survivors are flagged and never applied. Non-monotone re-entry (a candidate
   eliminated alone that appears in a surviving subset) is detected and reported.

### The grammar decides meaning, not the model

`grammar.py` declares, per operation and independently of any model output: the tuple slot it
edits, its admissibility tier (T1 intent-preserving / T2 intent-revealing / T3 intent-altering),
whether its parameters are BOUND or UNBOUND, and where a bound parameter comes from. A model
may propose a tier; the proposal is **recorded for audit and then overridden**. Classifying the
authority consequence of an edit is not delegated. `tier_agreed_with_model` in the report is
the audit signal — watch it.

---

## 5. The predicate is the load-bearing design decision

**Read this before changing `anomaly.py`.** The partition the cascade produces is exact only
if the predicate is, and a badly chosen predicate silently converts a physics question into an
artefact of the stress measure.

**Do not use "peak von Mises exceeds the allowable".** Holmberg Sec. 5 proves the clustered
P-norm *underestimates* the maximum local stress, and the paper states plainly that stresses
"will locally become higher than the stress limit". A peak predicate therefore fires on every
attainable design, and the only edit that entails it away is raising the allowable — which is
T3. **This was run and confirmed on this branch**: with a peak predicate, every physically
motivated candidate was eliminated, the sole survivor was `relax_stress_limit`, and the cascade
was driven into an authority refusal by the stress measure rather than by the structure. That
run is preserved as `artifacts/ato/ato_cascade_low_P3_peak_predicate.json` and is a result
worth keeping, not a mistake to hide.

**The discriminating quantity is the extent of overstress.** Measured at the `low` tier:

| | solid cells above 350 MPa | peak / p95 |
|---|---|---|
| P3 (stated problem) | 183 / 315 = **58%** | 1.12 — general overstress |
| P1 (stress-constrained) | 17 / 436 = **3.9%** | 1.33 — isolated local peaks |

`OVERSTRESS_FRACTION_LIMIT = 0.10` is a **declared calibration decision**, taken from the
paper's own criterion that a good design "only leaves a small number of points with stresses
above the stress limit". Any threshold between roughly 0.05 and 0.55 produces the same
partition, so the verdict is insensitive across an order of magnitude — state that whenever
the number is quoted. It was not tuned to make a particular candidate survive.

Three predicates are active: `widespread_overstress`, `reentrant_corner_overstress`,
`load_point_overstress`. Feature attribution reach is 2 filter radii, tied to the filter
because no feature smaller than the filter can be resolved.

### Known fidelity effect — report it, do not paper over it

At the `low` tier, P1 clears `widespread_overstress` but **still fires both feature
predicates**. The corner relief that separates P1 from P3 in the paper is partly suppressed
because `r0 = 1.5h = 7.5 mm` at this tier against 3 mm at reference — the filter is 2.5× coarser
relative to the domain, so the design cannot form fine corner relief. The consequence is a
**coupled defect**: no single edit clears all three targets, which is why `max_subset_size ≥ 2`
is not optional here. This is a genuine finding about how the framework propagates through a
lower-fidelity model, and it is one of the things this branch exists to measure.

---

## 6. Manifest status

**The hash-bound verification manifest is currently intact.** Everything added on this branch
lives in new modules (`project/topopt/stress/`, `project/ato/`,
`project/experiments/lbracket/`) that are not in `VERIFIED_RELATIVE_FILES`. `reference_q4.py`
and `optimization/mma.py` are *imported and reused, not modified* — the new solver shares the
repository's already verified Q4 element and the vendored Svanberg MMA rather than
re-implementing either.

Keep it that way where possible. If a hashed file must change, do it in **its own commit**,
re-run `python -m project.verification.run_suite`, and re-issue the manifest. Never leave the
tree with hashed files edited and the manifest stale.

DOLFINx is needed only for the original compliance path and the verification suite. The
stress-constrained solver and the whole cascade depend on **numpy and scipy only**.

---

## 7. Running things

Set threads to one first (see below), then:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# 0. validate the API key path -- one call, a few hundred tokens
python -m project.experiments.lbracket.run_ato --check-backend

# 1. adjoint verification -- after touching stress.py, solver.py or filters.py
python -m project.topopt.stress.verify_gradients

# 2. P1/P2/P3 benchmark at a tier, plus the bounded reference cost probe
python -m project.experiments.lbracket.run_benchmark --fidelity low

# 3. the reported cascade result -- NOTE --max-rounds 3; depth 1 abstains
python -m project.experiments.lbracket.run_ato \
    --fidelity low --max-candidates 5 --max-subset-size 3 \
    --discharge-max-iter 200 --max-rounds 3

# 4. predicate-set ablation: one candidate set, three target sets, cached re-solves
python -m project.experiments.lbracket.run_predicate_ablation --fidelity low

# 5. quantitative comparison against the published tables (see §9 -- it does not match)
python -m project.experiments.lbracket.run_paper_comparison --tier reference
```

`--max-rounds` defaults to 1. **The reported `flagged_intent_altering` result requires 3**; at
depth 1 the cascade correctly abstains with `no_survivor`. Requires `anthropic` and
`python-dotenv` on top of numpy/scipy.

Run as modules from the repository root. Outputs land in `artifacts/`.

**`.gitignore` gotcha.** `paths.py` sets `ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts"`, but
`.gitignore` only covers `Archive/artifacts/*`. A fresh top-level `artifacts/` is **not**
ignored. Extend `.gitignore` before committing, or generated PNGs and JSON will be staged.

### Threads: set them to one

**The solver is effectively serial — threaded BLAS buys nothing and costs a little.** Measured
on this machine: low-tier P3 takes 0.91 s at 1 thread and 0.93 s at 8; one reference-tier P1
iteration takes 0.118 s at 1 thread and 0.134 s at 8. The work is `scipy.sparse.linalg.splu`
plus the MMA subproblem, both serial, on matrices too small for threaded BLAS to amortise. A
cpu/wall ratio near the thread count (1825 s cpu against 228 s wall was observed) is OpenBLAS
threads **spin-waiting**, not parallel progress. Prefix runs with

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

which is marginally faster and leaves the rest of the machine free.

**The parallelism that would pay is process-level over discharges.** `run_discharge` re-solves
each candidate subset independently, so the loop is embarrassingly parallel; the 154-discharge
cascade at ~9 min serial would drop to roughly a minute across 8–16 worker processes. Not
implemented — the loop is sequential today.

### Cost control

Discharge count is `sum_{k=1..s} C(n, k)` for `n` candidates and subset size `s`:
`n=5, s=3` → 25 re-solves; `n=6, s=3` → 41. A P3 re-solve is ~1 s at the `low` tier; a P1
re-solve is ~15–35 s. Budget accordingly, and prefer reducing `max_candidates` over reducing
`max_subset_size` — the subset layer is where the non-monotone behaviour lives.

---

## 8. Working agreements

- **Small, attributable commits.** Solver, predicates, grammar, cascade, and any manifest
  re-issue are separate commits. A result that changes for two reasons at once cannot be
  attributed to either.
- **A candidate's verdict is the re-solve, not the argument.** If it was not re-solved, it has
  no verdict — record `NOT_ASSESSED`, never `ELIMINATED`.
- **Never hardcode an expected verdict**, and never tune a predicate threshold until a
  particular candidate survives. If a threshold is chosen, state its provenance and report the
  range over which the verdict is stable.
- **Record tier and boundedness for every candidate**, including eliminated ones (which carry
  no tier). Elimination does not require boundedness.
- **Enumerate the subset layer.** The coupled defect in §5 is not reachable from singletons.
- **Report the model's disagreements.** `tier_agreed_with_model` and the `rejected` list in the
  cascade report are the audit trail on generation quality; do not drop them from summaries.
- **Objective values are not comparable across formulation edits.** An edit that switches P3→P1
  changes the objective functional, so mass and compliance are not comparable across that
  boundary. The *stress field* remains comparable, which is why the predicates are defined on it.
- If a check was not run, write "not assessed". Do not estimate what it would have shown.

---

## 8a. The open cascade — architecture as of 2026-09-01

`project/ato/cascade.py` is the **closed** cascade: it hands the generator the edit grammar and
asks it to select. That caps the hypothesis space at whatever the toolbox contains, and the
symptom is that every proposal looks like parameter tuning. It is kept because the earlier
results in `RESULTS_L_bracket_ATO.md` §4 were produced with it; **do not use it for new work.**

`project/ato/open_cascade.py` is the current pipeline. Per epoch: gate → predicates →
**blind** hypothesis generation (`blind_hypothesize.py`, no catalogue in the payload) →
capability matching (`capability.py`) → deterministic coverage check (`coverage.py`) →
**mid-run tool synthesis** (`synthesize.py`) → discharge → ranking → authority
(`authority.py`) → accept. The accepted design becomes the next epoch's stated problem, so the
loop runs until every predicate is quiet or nothing further is authorised.

Invariants. Break any of these and the mechanism stops meaning anything:

1. **Tier is never delegated.** `grammar.tier_for_slot` assigns it from the slot and the
   declared intent. Measured: the writer's own tier proposal was wrong on 2 of 2 occasions
   where it offered one.
2. **Tier follows what the code touches, not what it declares.** `synthesize.changed_fields`
   diffs the specification and grades on the most restrictive slot touched. A writer bundled a
   model switch into a load-case operation; without this it would have carried the declared
   slot's tier.
3. **A synthesized operation may only mutate the specification.** AST whitelist plus a smoke
   test; no filesystem, network, `eval`, `getattr`, dunder access, or computation of a physical
   quantity. The reason is not tidiness: if written code could compute or assert anything, an
   agent could influence its own verdict.
4. **Verdicts come only from re-solving.** `NOT_REPRESENTABLE` is distinct from
   `NOT_ASSESSED` and from `ELIMINATED` — the question could not be posed at all.
5. **Authority comes from `intent_lbracket.json`, not from plausibility.**
   `authority._permitted_by_intent` decides independently whether adoption is available; a
   model recommendation to adopt something the intent does not cover is downgraded to
   escalate and the disagreement recorded.
6. **Synthesized operations are never written back to the committed grammar.** They live in
   `grammar.SYNTHESIZED` for the run; source is recorded in the run record.
   `project/ato/reference_ops.py` holds human-written equivalents, deliberately unregistered,
   as a control — registering them would stop the synthesis stage from ever being exercised.

Acceptance routes in `open_cascade.pick_edit_set`, in order: full survivor (T1/T2) →
authority-adopted T3 survivor → **disclosure step** (T1/T2 clearing ≥1 anomaly, permitted to
reveal new ones only from representation- or requirement-adding slots) → authority-adopted T3
partial repair. The disclosure route exists because demanding one edit set clear every fired
anomaly at once discards edits that demonstrably fixed one; the slot restriction exists
because a geometry or constraint edit that creates a new anomaly is a regression, not a
disclosure.

---

## 9. Results on record (2026-09-01)

Full write-up in **`RESULTS_L_bracket_ATO.md`**. Load it before re-running anything, so a
result is extended rather than silently contradicted. Raw records under `artifacts/`.

Established, with evidence:

- **Verification.** 6400 elements at h = 2 mm, matching the paper. Adjoint vs. central
  differences, directional: max relative error 1.5e-10 (P1), 9.6e-8 (P2, P3), with the error
  *growing* as the step shrinks. Manifest intact, 27 hashed files, none modified.
- **NOT a validated reproduction of the paper — qualitative agreement only.** P3 places material
  at the re-entrant corner and stress constraints move it out; that mechanism matches. The
  numbers do not. Measured at the paper's own 6400-element mesh
  (`run_paper_comparison.py`, `artifacts/benchmark/paper_comparison_reference.json`):
  P3 at the paper's exact mass (20.63 g) gives C = 5,003 against a published 10,960 — **2.2×
  too stiff** — and the formulation ordering is P1 < P3 < P2 here against a published
  P3 < P1 < P2. Every input was verified equal to the paper, including the objective definition
  C = ½Fᵀu which the paper states explicitly, and the solid L-beam mass is exactly 71.68 g, so
  the mesh/material/assembly are sound and the gap is unexplained. Untested candidates:
  continuation schedule, MMA move limits and asymptotes (unpublished), different local optimum.
  **Do not claim reproduction until this closes.**
- **P1 does not scale to the paper's mesh.** At 1024 elements it reaches 28.94 g stress-feasible;
  at 6400 elements over 400 iterations mass climbs to ~55 g and stays stress-infeasible near
  950 MPa peak. Each cluster then holds 640 points rather than 102 and the P-norm's underestimate
  of the maximum grows with cluster population. The reference tier is therefore a **cost probe
  only**, not a benchmark, until this is fixed. The §4 cascade results are unaffected (low tier
  throughout).
- **The proposal's flagship defect class is not exercised.** The worked example's defect is an
  omitted out-of-plane load case; 2-D plane stress cannot represent one at all (nodes carry only
  `u, v`). What ran here is a different class — jointly unsatisfiable stated requirements plus a
  geometric feature. Do not transfer conclusions between them.
- **Reference-tier cost on this machine:** 0.093–0.094 s/iteration (P3), 0.13–0.16 s/iteration
  (P1) at 6400 elements, across two runs. Single-sample wall-clock timings of one evaluation,
  not benchmarked means — re-measure rather than quoting a single figure.
- **The stated 0.30 mass budget is infeasible against the 350 MPa yield limit.** Established by
  a standalone plateau check, not by the discharge stage: the worst constraint plateaus flat at
  +0.21 at mass ≤ 0.30 over the last 20 of 400 iterations, and converges feasibly at mass ≤ 0.40.
  The paper says the same thing in prose for P2 at low allowable mass. **The cascade itself did
  not prove it**: of the 11 round-1 subsets involving `add_stress_constraints(P2)`, 6 returned
  INFEASIBLE and 5 returned NOT_ASSESSED, including the singleton (worst constraint still
  descending at the budget). Do not restate those NOT_ASSESSED discharges as infeasibility —
  that is the §8 rule, and this run is where it bites.
- **Coupled defect across three slots.** Each fired anomaly is cleared by an edit in a
  different slot: widespread overstress by `increase_mass_budget` (T3), corner overstress by
  `declare_corner_fillet(20 mm)` (T2), load-point overstress only by `relax_stress_limit` (T3).
  Terminal outcome `flagged_intent_altering` — nothing applied.
- **Non-monotone re-entry observed in 8 subsets.** `refine_stress_clusters` and
  `increase_mass_budget` were each eliminated alone and re-entered in surviving subsets.
- **Tier delegation would have broken the authority gate.** Of five candidates where the model
  volunteered a tier, one agreed with the grammar; it proposed **T1** for both
  `relax_stress_limit` and `increase_mass_budget`, which the grammar classifies **T3**. Keep
  tier assignment in the grammar.

Three standing cautions:

- **`AnthropicBackend` has never made a live call.** The recorded results came from an injected
  `CallableBackend` on platform model access (`Codex-sonnet-5`, 10 calls, 25624/6871 tokens);
  `ANTHROPIC_API_KEY` was absent from that environment and nothing was billed to it. The
  default backend in `run_ato.py` is the one that will spend real credits and only its
  `BackendError` guard is exercised — it needs `anthropic` and `python-dotenv` installed. Run
  `python -m project.experiments.lbracket.run_ato --check-backend` (one minimal call) before
  trusting a full cascade run, and do not report the backend as working until that passes.

- **`distribute_load` may be a spurious repair.** It clears `load_point_overstress`, but the
  paper (Sec. 9.4) warns that distributing the load yields a design more optimal with respect
  to the formulated problem yet physically useless, since a small load-direction perturbation
  would collapse it. No predicate in the current set catches this. A stiffness, buckling or
  eigenfrequency constraint — or a perturbed load case — is what would.
- **Do not treat the `low`-tier feature predicates as discriminating.** See §5. Testing the
  cascade at the `medium` tier is the open item; **not assessed**.