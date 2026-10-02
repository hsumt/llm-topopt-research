# 3-D L-bracket source

Imported on 2026-10-02 from Nelson Jurado's sibling `mfrto` working tree:
`p_norm_stress_constraint/lbracket/simplified_3D_holes/`.
The source repository HEAD was `18657db3e920861bb022572d58ea9aafc4396cb6`;
the file hashes below identify the actual working-tree inputs independently of HEAD.

| Original file | SHA-256 |
| --- | --- |
| `main.py` | `954dd17f47b5cf8221229d6ab14e11b12a806823247352d461a6865bc23faf88` |
| `utils.py` | `e78ee30d8a946404604a6e056afd90d6156ec7c5037e50515579b9816445e81e` |
| `pyfea.py` | `13d895036214dc7a43a489832c3992c117f9900fdde69ab745305a56bd387e50` |

The original main identifies A. Guibert, M. Pozzi and M. Bookwala as contributors
to the thermo-mechanical battery-pack example, and N. Jurado as modifier of the
L-bracket. Its five starting holes reference Kambampati, Chung & Kim (2021), Fig. 5;
the material and tip force reference Kambampati, Gray & Kim (2020), Sec. 3.1.
These are source-code attributions, not a claim of paper reproduction.
The three source files carried no separate license declaration; this port retains
their attribution and does not relicense them. ParaLeSTO is a distinct Apache-2.0
dependency; its bundled license governs that dependency.

## Integration changes

- `config.py` makes the physical inputs explicit in SI units and validates the
  equal-arm L, cubic mesh, five initial holes, material, force and constraints.
- `main.py` retains the original elasticity, compliance, normalized p-norm,
  sensitivities, level-set initialization and optimizer. It accepts JSON input,
  a fresh output directory and a bounded iteration budget. A vector resultant
  replaces the hardcoded downward force, and an explicit null stress limit
  selects volume-only optimization. The stress field is still evaluated.
- The original reference choices remain available only through the explicitly
  selected `reference_config()`. In particular, 116 MPa is the original calibrated
  p-norm limit, not a titanium yield strength.
- `pyfea.py` raises on every unsuccessful PETSc linear solve, including reason -8;
  the source merely printed a warning and continued.
- Sensitivity gathering uses owned degrees of freedom, and MPI ranks synchronize
  output creation. An unhandled rank failure aborts peers to prevent deadlock.
- `summary.json` records actual load, configuration, runtime, last evaluated
  metrics and output paths. Reaching the iteration budget is recorded as
  completed, with optimization convergence **not assessed**.

The top face of the vertical arm remains fully clamped, the top-right square
remains permanent void, and the tip load patch remains protected solid.
The five through-thickness holes are initial design voids: optimization may move,
merge or close them. Compliance is `Fᵀu`; the stress measure is
`[1/V_L ∫(rho * von_Mises)^p dV]^(1/p)` and is not a maximum-stress guarantee.

The installed reference runtime used for the local bounded integration probe is
Python 3.10, DOLFINx 0.7.0 and PyParaLeSTO 1.0.0. A probe demonstrates execution
and data plumbing; it does not establish optimizer convergence, a gradient
verification result, or validation against a published design.
