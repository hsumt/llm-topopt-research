"""
Deprecated 5/18/2026 OC update is heuristic and works well for a single vol constraint.
It isn't commonly used. Sigmund uses it as a learning example in 2001
"""
import numpy as np
def oc_update(rho, dc, volfrac, nelx, nely): # Sigmund 2001 Appendix (p. 126)
    l1, l2 = 0.0, 1e5
    move = 0.2

    while (l2 - l1) > 1e-4:
        lmid = 0.5 * (l1 + l2)

        rho_new = np.maximum(
            0.001,
            np.maximum(
                rho - move,
                np.minimum(
                    1.0,
                    np.minimum(
                        rho + move,
                        rho * np.sqrt(-dc / lmid)
                    )
                )
            )
        )

        if rho_new.sum() - volfrac * nelx * nely > 0:
            l1 = lmid
        else:
            l2 = lmid

    return rho_new