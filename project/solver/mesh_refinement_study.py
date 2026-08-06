"""Run the benchmark-level mesh-refinement V&V study.

The study keeps physical geometry, material, volume fraction, penalization, and
cone-equivalent physical ``r_min`` fixed while changing only the mesh. It does
not impose an arbitrary pass threshold; it records compliance and topology
changes for professor-approved interpretation.

Examples
--------
    /dolfinx-env/bin/python project/solver/mesh_refinement_study.py --case cantilever
    /dolfinx-env/bin/python project/solver/mesh_refinement_study.py --case mbb
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from project.solver._01_MSH._boundaries import (
    build_bcs,
    build_bcs_mbb,
    build_load,
    build_load_mbb,
)
from project.solver._01_MSH._domain import get_lame_parameters
from project.solver._01_MSH._mesh import build_mesh
from project.solver._02_FEA._functionspaces import build_spaces
from project.solver._04_PPRCS.postprocess import build_cell_perm
from project.solver._verification_manifest import source_hashes
from project.solver.SIMP_MASTER import CASE_PARAMS, _run_optimization


DEFAULT_MESHES = {
    "cantilever": [(40, 25), (80, 50), (160, 100)],
    "mbb": [(60, 20), (120, 40), (240, 80)],
}


def _parse_meshes(text: str) -> list[tuple[int, int]]:
    result = []
    for item in text.split(","):
        nx, ny = item.lower().split("x", maxsplit=1)
        pair = (int(nx), int(ny))
        if min(pair) <= 0:
            raise ValueError("mesh dimensions must be positive")
        result.append(pair)
    if len(result) < 2:
        raise ValueError("mesh-refinement study requires at least two meshes")
    return result


def _nearest_resample(grid: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    sy, sx = grid.shape
    ty, tx = target_shape
    iy = np.clip(
        np.floor((np.arange(ty) + 0.5) * sy / ty).astype(int), 0, sy - 1
    )
    ix = np.clip(
        np.floor((np.arange(tx) + 0.5) * sx / tx).astype(int), 0, sx - 1
    )
    return grid[np.ix_(iy, ix)]


def _build_case(case: str, nelx: int, nely: int, output_dir: Path):
    base = dict(CASE_PARAMS[case])
    base["nelx"] = int(nelx)
    base["nely"] = int(nely)
    base["element_size"] = min(base["Lx"] / nelx, base["Ly"] / nely)
    base["r_min_convention"] = "cone_equivalent_radius"
    base["r_pde"] = float(base["r_min"]) / (2.0 * np.sqrt(3.0))
    base["r_min_elements"] = float(base["r_min"]) / base["element_size"]
    mu, lmbda = get_lame_parameters(base["E"], base["nu"])
    base.update(mu=mu, lmbda=lmbda)

    domain = build_mesh(nelx, nely, base["Lx"], base["Ly"])
    V, Q = build_spaces(domain)
    if case == "cantilever":
        bcs = build_bcs(V, domain)
        load = build_load(V, domain, base["Lx"], base["Ly"], nely)
    else:
        bcs = build_bcs_mbb(V, domain, base["Lx"], base["Ly"])
        load = build_load_mbb(V, domain, base["Lx"], base["Ly"], nely)
    perm = build_cell_perm(domain, Q, nelx, nely, base["Lx"], base["Ly"])
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    result = _run_optimization(
        domain=domain,
        V=V,
        Q=Q,
        bcs=bcs,
        F_load=load,
        p=base,
        frames_dir=str(frames_dir),
        perm=perm,
        save_frames=False,
    )
    grid = result.rho_final[perm].reshape((nely, nelx))
    np.savez_compressed(
        output_dir / "final_state.npz",
        rho_phys=grid,
        compliance=np.array(result.metrics["compliance_history"][-1]),
    )
    return base, result, grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(DEFAULT_MESHES), default="cantilever")
    parser.add_argument(
        "--meshes",
        help="comma-separated meshes, e.g. 40x25,80x50,160x100",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "solver" / "_05_OUT" / "vv"),
    )
    args = parser.parse_args()

    meshes = _parse_meshes(args.meshes) if args.meshes else DEFAULT_MESHES[args.case]
    output_root = Path(args.output)
    study_root = output_root / f"mesh_refinement_{args.case}"
    if study_root.exists():
        shutil.rmtree(study_root)
    study_root.mkdir(parents=True)

    records = []
    grids = []
    for nelx, nely in meshes:
        run_dir = study_root / f"{nelx}x{nely}"
        run_dir.mkdir(parents=True)
        print(f"\n=== Mesh-refinement run {nelx}x{nely} ===")
        p, result, grid = _build_case(args.case, nelx, nely, run_dir)
        grids.append(grid)
        records.append(
            {
                "nelx": nelx,
                "nely": nely,
                "Lx": p["Lx"],
                "Ly": p["Ly"],
                "element_size": p["element_size"],
                "r_min": p["r_min"],
                "r_min_elements": p["r_min_elements"],
                "iterations": result.metrics["iteration"],
                "design_converged": result.metrics["design_converged"],
                "objective_plateau": result.metrics["objective_plateau"],
                "continuation_complete": result.metrics["continuation_complete"],
                "final_compliance": result.metrics["compliance_history"][-1],
                "final_volume_fraction": float(
                    np.dot(result.rho_final, result.cell_volumes)
                    / result.cell_volumes.sum()
                ),
                "final_change": result.metrics["change_history"][-1],
            }
        )

    finest_shape = grids[-1].shape
    finest = grids[-1]
    for record, grid in zip(records, grids):
        mapped = _nearest_resample(grid, finest_shape)
        record["mean_abs_density_difference_to_finest"] = float(
            np.mean(np.abs(mapped - finest))
        )
    for i, record in enumerate(records):
        if i == 0:
            record["compliance_change_from_previous"] = None
        else:
            previous = records[i - 1]["final_compliance"]
            record["compliance_change_from_previous"] = float(
                abs(record["final_compliance"] - previous)
                / max(abs(record["final_compliance"]), 1.0e-30)
            )

    manifest = {
        "schema_version": 1,
        "completed": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "case": args.case,
        "interpretation": (
            "No universal pass threshold is imposed. Review compliance changes, "
            "density-field differences, convergence status, and the discrete "
            "point-load singularity qualification with the project supervisor."
        ),
        "fixed_physical_parameters": {
            "Lx": CASE_PARAMS[args.case]["Lx"],
            "Ly": CASE_PARAMS[args.case]["Ly"],
            "r_min": CASE_PARAMS[args.case]["r_min"],
            "volfrac": CASE_PARAMS[args.case]["volfrac"],
            "penal": CASE_PARAMS[args.case]["penal"],
            "point_load_qualification": (
                "global discrete compliance is compared; pointwise stress near "
                "the loaded node is not treated as mesh convergent"
            ),
        },
        "records": records,
        "source_hashes": source_hashes(PROJECT_ROOT),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "mesh_refinement_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nMesh-refinement manifest written: {manifest_path}")


if __name__ == "__main__":
    main()
