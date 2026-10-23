"""Dry run: ParaLeSTO builds the starting design from a config, in a child process, before approval.

The solver's first step (main.run: build_level_set, then calculate_element_densities in iteration 0)
runs here on its own, and the result is compared with the geometry the config states:
- ParaLeSTO's native code can fail on accepted configs (SIGSEGV, or exit 1 "There are no frozen
  nodes!") when an initial hole overlaps the solid square under the load or leaves a gap of about
  one element to an edge, the cut or another hole;
- it can also run but start from a different geometry: a load square that is not solid, or a gap
  narrower than the mesh can show, opened or closed.
Either way a run from that config would not solve what was stated, so the code check reports it
(lbracket.start_issues). A fresh process per config keeps a crash out of the app's process, and out
of ParaLeSTO's reach of the app's old heap memory (LevelSet::initialise reads uninitialised hole
fields). Outcomes are kept per config: the child is deterministic for its input.

    python -m project.solver.lbracket2d.dryrun < config.json   (prints the child's JSON result)
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .config import LBracket2DConfig

VOID = .25         # an element of the starting design is void below this density (above it, weak material)
SOLID = .999       # minimum density of an element inside the load square
TIMEOUT_S = 20     # the child normally finishes in about 0.1 s


@dataclass(frozen=True)
class DryRun:
    densities: list | None            # (nely, nelx) element area fractions, rows bottom to top; None if it failed
    failure: str | None               # why ParaLeSTO stopped: "signal 11", "exit code 1: <last stderr line>", ...
    problems: tuple[tuple[str, str], ...] = ()   # (spec field, description): where the start differs from the config
    transient: bool = False           # the child could not be launched or did not finish: no verdict, retry later

    @property
    def ok(self) -> bool:
        return self.failure is None and not self.problems


def _stated_regions(config: LBracket2DConfig) -> tuple[list[int], int]:
    """Which stated initial holes form one opening, and which reach the outside (an outer edge or the cut),
    exactly: holes join when they overlap or touch (centre distance <= 2r), and a hole opens when it reaches
    an edge or the solver's rounded cut (distance <= r). Returns a region id per hole and the outside's id."""
    import math

    nelx, nely = config.mesh_shape
    cells = nelx / config.lx_m   # elements per metre
    arm = nelx - config.cut_cells   # the cut is x > arm and y > arm (lx == ly)
    r = config.hole_radius_m * cells
    centres = [(x * cells, y * cells) for x, y in config.hole_centers_m]
    outside = len(centres)
    parent = list(range(outside + 1))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, (x, y) in enumerate(centres):
        if min(x, y, nelx - x, nely - y, math.hypot(max(arm - x, 0), max(arm - y, 0))) <= r:
            parent[find(i)] = find(outside)
        for j in range(i + 1, len(centres)):
            if math.dist(centres[i], centres[j]) <= 2 * r:
                parent[find(i)] = find(j)
    return [find(i) for i in range(len(centres))], find(outside)


def _void_regions(void):
    """Labels of the 8-connected void regions of a bool grid framed by a void border (the outside):
    an int array the size of the framed grid, 0 for material, and the label of the outside."""
    import numpy as np
    from collections import deque

    framed = np.pad(void, 1, constant_values=True)
    labels = np.zeros(framed.shape, dtype=np.int32)
    rows, cols = framed.shape
    count = 0
    for start in zip(*np.nonzero(framed)):
        if labels[start]:
            continue
        count += 1
        labels[start] = count
        queue = deque([start])
        while queue:
            i, j = queue.popleft()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a, b = i + di, j + dj
                    if 0 <= a < rows and 0 <= b < cols and framed[a, b] and not labels[a, b]:
                        labels[a, b] = count
                        queue.append((a, b))
    return labels, labels[0, 0]


def problems(config: LBracket2DConfig, densities) -> list[tuple[str, str]]:
    """Where ParaLeSTO's starting design differs from the stated geometry, as (spec field, description):
    the load square not solid, or an initial hole that closes, opens to the outside or the cut, merges
    with another hole, or separates from one it overlaps. Shape detail inside one region (a rounded cusp
    between overlapping holes) does not count."""
    import numpy as np

    densities = np.asarray(densities, dtype=float)
    nelx, nely = config.mesh_shape
    h_mm = config.lx_m / nelx * 1e3
    arm, patch = nelx - config.cut_cells, config.patch_cells
    found = []
    square = densities[arm - patch:arm, nelx - patch:nelx]   # the solid square under the load
    if patch and square.min() < SOLID:
        found.append(("geometry.parameters.hole_centers",
                      f"The solid square under the load (the last {patch * h_mm:g} mm of the tip band) is not solid in "
                      f"ParaLeSTO's starting design (lowest element density {square.min():.2f}): an initial hole reaches "
                      "into it. Keep the holes clear of the loaded square at the tip."))
    s, stated_out = _stated_regions(config)
    start, start_out = _void_regions(densities < VOID)
    cells = nelx / config.lx_m
    centres = [(cx * cells, cy * cells) for cx, cy in config.hole_centers_m]   # in elements
    # A hole's void in the starting design: the emptiest void element within its stated footprint (the level set
    # can shift a small hole off its centre element); 0 when none is void, i.e. the hole closes.
    radius = config.hole_radius_m * cells
    cols, rows = np.meshgrid(np.arange(nelx) + .5, np.arange(nely) + .5)
    p = []
    for cx, cy in centres:
        near = ((cols - cx) ** 2 + (rows - cy) ** 2 <= (radius + .5) ** 2) & (densities < VOID)
        if near.any():
            i, j = np.unravel_index(np.where(near, densities, np.inf).argmin(), densities.shape)
            p.append(start[i + 1, j + 1])
        else:
            p.append(0)
    mm = lambda k: f"hole {k + 1} at ({config.hole_centers_m[k][0] * 1e3:g}, {config.hole_centers_m[k][1] * 1e3:g}) mm"
    changes = [f"{mm(k)} closes" for k in range(len(centres)) if not p[k]]
    changes += [f"{mm(k)} opens to the outside or the cut" for k in range(len(centres))
                if p[k] and s[k] != stated_out and p[k] == start_out]
    for k in range(len(centres)):
        for m in range(k + 1, len(centres)):
            if p[k] and p[m] and s[k] != s[m] and p[k] == p[m] and p[k] != start_out:
                changes.append(f"holes {k + 1} and {m + 1} merge")
    names = lambda group: f"hole {group[0]}" if len(group) == 1 else "holes " + ", ".join(map(str, group))
    regions: dict[int, list[int]] = {}   # stated opening -> its holes that stay open
    for k in range(len(centres)):
        if p[k]:
            regions.setdefault(s[k], []).append(k)
    for members in regions.values():
        parts: dict[int, list[int]] = {}   # opening in the starting design -> hole numbers
        for k in members:
            parts.setdefault(p[k], []).append(k + 1)
        if len(parts) > 1:
            changes.append(f"{names([k + 1 for k in members])}, which overlap as one opening, split into "
                           + " and ".join(names(group) for group in parts.values()))
    if changes:
        found.append(("geometry.parameters.hole_centers",
                      "ParaLeSTO's starting design differs from the stated geometry: " + "; ".join(changes)
                      + f". On this mesh ({h_mm:g} mm elements) the level set cannot show a hole of radius about one element "
                      "or less, or a gap of about one element or less between a hole and an edge, the cut or another hole."))
    return found


_outcomes: dict[str, DryRun] = {}
_retries: dict[str, tuple[float, DryRun]] = {}   # no verdict yet (launch error, timeout): kept a few seconds
RETRY_S = 5


def dry_run(config: LBracket2DConfig) -> DryRun:
    """ParaLeSTO's starting design for a config, built in a fresh child process, and how it differs
    from the stated geometry. Kept per config (bounded); a timeout or launch error only for RETRY_S."""
    key = json.dumps(config.to_dict(), sort_keys=True)
    if key in _outcomes:
        return _outcomes[key]
    if key in _retries and time.monotonic() - _retries[key][0] < RETRY_S:   # one page render launches it once
        return _retries[key][1]
    from project.execution import MODEL_CREDENTIALS   # imported here: the solver package does not need the app

    root = Path(__file__).resolve().parents[3]
    environment = {k: v for k, v in os.environ.items() if k not in MODEL_CREDENTIALS}
    environment["PYTHONPATH"] = os.pathsep.join(p for p in (str(root), environment.get("PYTHONPATH")) if p)
    try:
        result = subprocess.run([sys.executable, "-m", __name__], input=key, capture_output=True, text=True,
                                cwd=root, env=environment, timeout=TIMEOUT_S, check=False)
    except (subprocess.TimeoutExpired, OSError) as error:
        reason = f"it did not finish in {TIMEOUT_S} s" if isinstance(error, subprocess.TimeoutExpired) else type(error).__name__
        _retries[key] = time.monotonic(), DryRun(None, reason, transient=True)
        return _retries[key][1]
    if result.returncode == 0:
        out = json.loads(result.stdout)
        outcome = DryRun(out["densities"], None, tuple(map(tuple, out["problems"])))
    else:
        last = next((line.strip() for line in reversed(result.stderr.splitlines()) if line.strip()), "")
        code = f"signal {-result.returncode}" if result.returncode < 0 else f"exit code {result.returncode}"
        outcome = DryRun(None, code + (f": {last[:160]}" if last else ""))
    if len(_outcomes) >= 64:   # bounded: drop the oldest
        _outcomes.pop(next(iter(_outcomes)), None)
    _outcomes[key] = outcome
    return outcome


def _child() -> None:
    from .levelset import initial_densities

    config = LBracket2DConfig.from_dict(json.load(sys.stdin))
    densities, _ = initial_densities(config)
    json.dump({"densities": densities.round(3).tolist(), "problems": problems(config, densities)}, sys.stdout)


if __name__ == "__main__":
    _child()
