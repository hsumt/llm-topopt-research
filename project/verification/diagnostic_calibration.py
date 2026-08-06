"""Generate deterministic baseline values for spatial quality diagnostics.

This script does not choose a checkerboard pass threshold. It demonstrates how
the alternating-mode proxy responds to synthetic fields so the project can
calibrate or redesign the metric using professor-approved examples.


    /dolfinx-env/bin/python -m project.verification.diagnostic_calibration


"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from project.verification.gate import _checkerboard_index
from project.verification.manifest import source_hashes
from project.paths import PROJECT_ROOT, VV_ARTIFACT_ROOT

def _horizontal_member(ny=40, nx=120):
    grid = np.zeros((ny, nx), dtype=float)
    grid[ny // 2 - 2 : ny // 2 + 2, :] = 1.0
    return grid


def _diagonal_member(ny=40, nx=120, half_width=2.0):
    y, x = np.indices((ny, nx))
    center = (ny - 1) * x / max(nx - 1, 1)
    return (np.abs(y - center) <= half_width).astype(float)


def _smooth_diagonal(ny=40, nx=120, width=2.0):
    y, x = np.indices((ny, nx))
    center = (ny - 1) * x / max(nx - 1, 1)
    distance = np.abs(y - center)
    return np.clip(1.0 - distance / max(width, 1.0e-12), 0.0, 1.0)


def main():
    ny, nx = 40, 120
    fields = {
        "uniform": np.full((ny, nx), 0.4),
        "perfect_checkerboard": (np.indices((ny, nx)).sum(axis=0) % 2).astype(float),
        "horizontal_member": _horizontal_member(ny, nx),
        "stair_step_diagonal_member": _diagonal_member(ny, nx),
        "smooth_diagonal_member": _smooth_diagonal(ny, nx),
        "vertical_stripes": (np.indices((ny, nx))[1] % 2).astype(float),
    }
    records = {
        name: {
            "checkerboard_alternating_mode_index": _checkerboard_index(grid),
            "gray_fraction_0p25_0p75": float(
                np.mean((grid > 0.25) & (grid < 0.75))
            ),
        }
        for name, grid in fields.items()
    }
    payload = {
        "schema_version": 1,
        "completed": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "Baseline values only. No Boolean checkerboard threshold is "
            "established by this synthetic calibration set."
        ),
        "records": records,
        "source_hashes": source_hashes(PROJECT_ROOT),
    }
    out = VV_ARTIFACT_ROOT / "diagnostic_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(records, indent=2))
    print(f"Calibration baseline written: {out}")


if __name__ == "__main__":
    main()
