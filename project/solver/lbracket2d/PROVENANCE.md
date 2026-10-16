# Provenance

Ported 2026-10-15 from `mfrto/p_norm_stress_constraint/lbracket/simplified_2D_holes`
(committed in mfrto 68a97f8; last edited 2 Oct). It replaces the provisional 3-D
port in `project/solver/lbracket3d` as the connected backend.

- `main.py`: the source script wrapped in `run(config, output, max_iterations)`. Inputs
  come from `LBracket2DConfig` instead of module constants; output goes to a new run
  folder with `input.json` and `summary.json`; the start-geometry assertion (valid only
  for the reference geometry) now records the start in the summary instead; the
  stress constraint is optional (volume only when `stress_limit_pa` is None).
  Physics, level-set setup and optimizer calls are unchanged.
- `pyfea.py`, `utils.py`: copied unchanged.
- `config.py`: adapted from the 3-D port's validated contract (plane stress, square
  cells, a 2-component force). Reference values are the source's, with the 0.75
  volume fraction of its 2 Oct vf75 run and the source's 125 MPa calibrated limit.

Not ported: `KCK_2021_Replication_2D` (8 Oct, untracked) is unfinished
(`vol_fraction` and `Sp_limit` undefined; expects a patched ParaLeSTO).
