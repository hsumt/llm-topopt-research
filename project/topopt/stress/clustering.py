"""Distribution of stress evaluation points into clusters, Holmberg Sec. 6.

Two techniques are implemented, exactly as described in the paper:

* ``stress_level``       -- sort descending, fill cluster 1 with the first
                            ne/nc points, cluster 2 with the next, and so on
                            (Eq. 8).  Gives the best local stress control.
* ``distributed_stress`` -- sort descending, then deal points round-robin, so
                            every cluster spans the whole stress range.

The paper's conclusion is that stress-level clustering with reclustering every
iteration is preferable; that is the default here.
"""
from __future__ import annotations

import numpy as np

TECHNIQUES = ("stress_level", "distributed_stress")


def assign_clusters(sigma_vm: np.ndarray, n_clusters: int, technique: str) -> list[np.ndarray]:
    if technique not in TECHNIQUES:
        raise ValueError(f"unknown clustering technique {technique!r}; choose from {TECHNIQUES}")
    n = sigma_vm.size
    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")
    n_clusters = min(n_clusters, n)
    order = np.argsort(-sigma_vm, kind="stable")

    if technique == "stress_level":
        per = int(np.ceil(n / n_clusters))
        groups = [order[i * per:(i + 1) * per] for i in range(n_clusters)]
    else:
        groups = [order[i::n_clusters] for i in range(n_clusters)]

    groups = [g for g in groups if g.size > 0]
    if not groups:
        raise RuntimeError("clustering produced no non-empty cluster")
    return groups
