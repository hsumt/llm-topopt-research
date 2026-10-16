# Provenance

Ported 2026-10-15 from `mfrto/p_norm_stress_constraint/lbracket/simplified_2D_holes`
(committed in mfrto 68a97f8; last edited 2 Oct). It replaces the provisional 3-D
port that was in `project/solver/lbracket3d` as the connected backend; that port has
been deleted.

- `main.py`: the source script wrapped in `run(config, output, max_iterations)` with a
  command-line entry point. Changes from the source:
  - inputs come from `LBracket2DConfig` instead of module constants;
  - output goes to a new run folder with `input.json` and `summary.json`, instead of a
    timestamped folder under mfrto's `benchmarks/results`;
  - the load is a configured in-plane force vector (Fx, Fy) instead of the downward
    `F_tip`, and the applied-load check integrates the traction magnitude |T| instead
    of its downward component;
  - the start-geometry assertion (valid only for the reference geometry) now records
    the start design in the summary; the check for an empty start remains;
  - the stress constraint is optional: without `stress_limit_pa` the optimizer gets the
    volume row only;
  - the run also writes `states/density_final.npy` and `states/density_history.npz`
    (element densities every 10 iterations and at the last iteration);
  - the per-iteration rebuild of the forms `L` and `a`, which were never recompiled,
    and the commented-out binary restart are removed;
  - the level-set initialization moved to `levelset.build_level_set`, and the element
    counts come from `LBracket2DConfig`, whose validation replaces the source's
    square-cell and whole-element assertions.

  The FE model, level-set setup and optimizer calls are otherwise unchanged.
- `pyfea.py`, `utils.py`: copied unchanged apart from a provenance docstring.
- `config.py`: new; adapted from the 3-D port's validated contract (equal outer
  lengths, so square cells; plane-stress thickness; a 2-component force), with the
  element counts (`mesh_shape`, `cut_cells`, `patch_cells`) in the source's formulas.
  Reference values are the source's, with the 0.75 volume fraction of its 2 Oct vf75
  run and the source's 125 MPa calibrated limit.
- `levelset.py`: new; `build_level_set` holds the source's level-set initialization
  (ParaLeSTO alone, no FEM). `main.run` and the formulation preview
  (`initial_densities`) both call it.
- `render_paraview.py`: new; pvbatch element renderer that `project/execution.py` runs
  after a successful solve.

Not ported: `KCK_2021_Replication_2D` (8 Oct, untracked) is unfinished
(`vol_fraction` and `Sp_limit` undefined; expects a patched ParaLeSTO).
