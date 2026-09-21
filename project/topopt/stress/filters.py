"""Design-variable cone filter, Holmberg et al. (2013) Sec. 3, Eq. (2).

rho_e = sum_j W_ej x_j with cone weights w_j = (r0 - r_j) / r0 over element
centroids within r0, normalized row-wise.  This is the *design variable* filter
of Bruns and Tortorelli (2001), not the Helmholtz PDE filter used by the
repository's compliance path; the paper's stress penalization and cluster
sensitivities are written against this form.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


def cone_filter_matrix(centroids: np.ndarray, r0: float) -> sp.csr_matrix:
    if r0 <= 0.0:
        raise ValueError("filter radius r0 must be positive")
    tree = cKDTree(centroids)
    pairs = tree.query_pairs(r0, output_type="ndarray")
    i = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(len(centroids))])
    j = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(len(centroids))])
    d = np.linalg.norm(centroids[i] - centroids[j], axis=1)
    w = np.maximum(r0 - d, 0.0) / r0
    W = sp.coo_matrix((w, (i, j)), shape=(len(centroids), len(centroids))).tocsr()
    rowsum = np.asarray(W.sum(axis=1)).ravel()
    if np.any(rowsum <= 0.0):
        raise RuntimeError("cone filter produced a zero row")
    return sp.diags(1.0 / rowsum) @ W
