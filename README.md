# 3D L-bracket formulation and level-set optimization

This branch connects the engineering pre-solve conversation to the five-hole
`simplified_3D_holes` L-bracket from MFRTO. Describe the physical problem, answer
missing engineering decisions, review the explicit solver inputs, and launch an
isolated FEniCSx/PyParaLeSTO run. Python checks the solver contract independently
of the model's review.

## Start the clean Docker environment

Docker Engine with Compose is the only host dependency. The image contains the
application, pinned numerical environment, and ParaLeSTO compiled from the
vendored `_external/pyparalesto` sources. It does not use a host Conda environment,
a sibling repository, or a compiled library from the host.

```bash
./docker/compose up --build -d
```

Open [the application](http://localhost:8502). The wrapper creates the output
folder under your user ID and disables implicit `.env` loading. Port 8502 avoids
conflicting with the earlier demo; override it with `LBRACKET_PORT` if needed.
To open a shell in the environment:

```bash
./docker/compose exec lbracket bash
```

Natural-language parsing and clarification use `ANTHROPIC_API_KEY` already set in
the invoking shell's environment. Set it through your normal credential manager,
then recreate the service with `./docker/compose up -d --force-recreate`. No
credential belongs in the image, repository, command arguments, or run receipts.
`FORMULATION_MODEL` optionally changes the configured model. The **Load complete
3D benchmark** button and command-line benchmarks require no model credential.

Application code is baked into the image. Rebuild after code changes. Generated
runs and saved formulations persist under `artifacts/`; the named home volume
holds JIT caches. Stop with `./docker/compose down`; this preserves both.

## Benchmark and test

A bounded smoke run exercises the complete specification adapter, approval
receipt, separate solver worker, and 3D finite-element solve:

```bash
./docker/compose run --rm lbracket python -m project.execution --reference --max-iterations 2 --ranks 1
./docker/compose run --rm lbracket python -m project.execution --reference --max-iterations 2 --ranks 2
./docker/compose run --rm lbracket python -m unittest discover -s tests -v
```

The numerical stack is DOLFINx **0.7.0**, FFCx/Basix **0.7.0**, UFL **2023.2.0**,
PETSc/petsc4py **3.19.6**, MPICH, NumPy **1.26.4**, and PyParaLeSTO **1.0.0**.
`docker/conda-linux-64.lock` fixes the exact public conda-forge builds and package
hashes; the base image is digest-pinned. This is a Linux x86-64 numerical image.
The solver uses the 0.7 APIs (`VectorFunctionSpace`, PETSc-backed function vectors),
not legacy FEniCS 2019 or an unpinned `dolfinx:stable` image. All BLAS/OpenMP thread
counts default to one. MPI processes supply the explicit parallelism.

The default reference problem has:

| Input | Value |
|---|---|
| Bounding box | 100 × 100 × 12 mm |
| Permanent cut | 60 × 60 mm, upper-right corner, through thickness |
| Initial holes | Five radius-10 mm cylinders; centers (20,20), (20,50), (20,80), (50,20), (80,20) mm |
| Material | Uniform isotropic elasticity, E = 120 GPa, ν = 0.36 |
| Support | All three translations fixed on the top face of the vertical arm |
| Load | Total [0, −5000, 0] N uniformly distributed on the 4 mm tip band through thickness |
| Objective | Minimum compliance, **Fᵀu** |
| Volume limit | 75% of the L domain, excluding the permanent square cut |
| Stress constraint | Density-weighted, volume-averaged p-norm, p = 6, ≤116 MPa |
| Mesh | 50 × 50 × 6 cubic hexahedra in the bounding box |

The holes are **initial design voids**: they may move, merge, or close. The
116 MPa limit is calibrated for this aggregate measure; it is **not titanium's
yield stress or a bound on peak von Mises stress**. This 3D example is an extension,
not a validated reproduction of a published 3D bracket.

The recorded iteration-zero calibration is approximately volume fraction
0.7527603, compliance 2.875124 J, and p-norm stress 107.1554 MPa. The initial design
slightly exceeds its volume constraint. A one- or two-iteration run checks
execution, not convergence or final feasibility. The source uses a fixed iteration
budget and no optimization convergence criterion; run receipts say convergence
**not assessed**. Use up to 500 iterations only when you intend a longer solve.

## Specification and conversation contract

`ProblemSpec` remains the user-facing physical description. The parser extracts
only supplied decisions. The reviewer sees the original request, context, current
specification and prior answers. Deterministic missing-decision checks are combined
with the model's review, with at most four questions per round. Missing material
constants are data requirements; no nominal constants are invented. Additional
information and authoritative data can be supplied through the correction box or
by importing an edited, validated formulation JSON.

`project/lbracket.py` accepts an explicit `lbracket3d_five_holes` geometry template,
unit-bearing dimensions and hole coordinates, a uniform material, the top clamp,
and one total tip force with any explicit x/y/z components. It converts supported
units to SI and constructs `LBracket3DConfig`. A volume-only problem requires an
explicit `stress_requirement="volume_only"`; a p-norm constraint requires its own explicit
limit. No benchmark physical defaults are silently inserted into custom requests.

Other geometries, fixed mounting holes, fillets, other supports, multiple load
cases, peak/yield constraints, nonlinear/dynamic physics and manufacturing
constraints are reported as unsupported. They are not silently removed or
reinterpreted. The current implementation does not import CAD. Dimensions must
admit the backend's cubic grid and at least six cells through each direction.
Numerical settings belong to the adapter/backend, not the engineering interview.

Before execution, the app shows the concrete SI configuration and iteration/MPI
budget. Approval is bound to their hash; changes require a fresh approval. The
model cannot authorize a solve. `--reference` explicitly selects the entire
reference configuration; `--session /app/artifacts/sessions/<id>.json` runs a saved,
ready formulation through the same checks.

Each run under `artifacts/runs/lbracket3d/` contains:

- `session.json`: request, context, review, questions, answers and typed changes.
- `config.json` and `approval.json`: exact inputs, run budget, hash and approval time.
- `status.json`, `worker.log` and `solver.log`: execution status and diagnostics.
- `solver/summary.json`, `convergence.txt`, `timings.txt`, STL surfaces, and
  XDMF/HDF5 density, displacement and stress fields for ParaView.

Saved sessions can be resumed or downloaded/imported. Completed runs survive
browser reloads in the artifact directory; interrupted workers are distinguished
from completed solves. The final reported metrics and saved fields describe the
last **evaluated** design, before its subsequent level-set update.

## Source and research context

- [Context from Claude chats and presentations](docs/3d_lbracket_context.md)
- [Solver source provenance and port changes](project/solver/lbracket3d/PROVENANCE.md)
- [Vendored ParaLeSTO source inventory](_external/pyparalesto-source.json)

The older 2D Holmberg/cascade material in local context and `Archive/` is historical
and has not been resurrected. This branch does not run the old post-solve cascade.

## Integration checks recorded on 2026-10-02

The clean image built successfully from the vendored source. All **32** unit tests
passed inside it. Streamlit's application test exercised the approval gate and
launched the second run below through the actual Run button.

| Bounded Docker run | Last evaluated volume fraction | Compliance Fᵀu, J | p6 stress, MPa |
|---|---:|---:|---:|
| Reference, 1 MPI rank, 2 iterations | 0.74575255 | 2.83664529 | 106.320756 |
| Volume-only, force [0, −5000, 250] N, 2 MPI ranks, 1 iteration | 0.75276026 | 2.95401947 | 109.382644 |

Both runs integrated the requested force to within 1e-6 N and wrote every declared
output. The first was feasible against its specified volume and p-norm limits at
the last evaluated design. The second still exceeded its volume bound; it had no
stress constraint. These are short execution checks, not converged optimization
results. New gradient verification, paper validation, and live model-provider
calls are **not assessed** by these checks.
