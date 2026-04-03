"""
agents.py — Physics gate + LLM agents for SIMP topology optimisation
=====================================================================
Three components:

1. PhysicsGate       — hard assertion checks (compliance, vol, checkerboard)
2. SteeringAgent     — calls Claude every N iters, adjusts p / r_min / volfrac
3. CriticAgent       — reads final metrics, returns natural language summary

Usage (standalone test):
    python agents.py
"""

import json
import re
import numpy as np
import requests


# ══════════════════════════════════════════════════════════════════════════════
# 1. PHYSICS ASSERTION GATE
# ══════════════════════════════════════════════════════════════════════════════
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env into os.environ
API_KEY = os.getenv("ANTHROPIC_API_KEY")
class PhysicsGate:
    """
    Hard physics checks run after every iteration (or on demand).

    Returns (passed: bool, reasons: list[str]) from .check().
    Call .assert_all() to raise AssertionError on first failure.

    Checks
    ------
    - compliance_non_increasing : C[i] <= C[i-1] + tol  (allow small rises)
    - vol_frac_within_tol       : |mean(rho) - target| < vol_tol
    - no_checkerboard           : local density variance pattern test
    """

    def __init__(
        self,
        volfrac_target: float,
        vol_tol: float = 0.02,
        compliance_rise_tol: float = 0.10,   # allow 10% rise (continuation phase)
        checkerboard_threshold: float = 0.4,  # max allowed checkerboard score
    ):
        self.volfrac_target        = volfrac_target
        self.vol_tol               = vol_tol
        self.compliance_rise_tol   = compliance_rise_tol
        self.checkerboard_threshold = checkerboard_threshold

    # ── individual checks ──────────────────────────────────────────────────

    def check_compliance(self, compliance_history: list[float], itr: int = 0, penal_steps: int = 0) -> tuple[bool, str]:
        """
        Compliance should be non-increasing once penalisation is complete.
        """
        # --- NEW: Suppress during the ramp phase ---
        if itr < penal_steps:
            return True, f"compliance_non_increasing: SKIP (ramp phase, itr {itr} < {penal_steps})"

        if len(compliance_history) < 2:
            return True, "compliance_non_increasing: not enough data (SKIP)"

        c_prev = compliance_history[-2]
        c_curr = compliance_history[-1]

        if c_prev <= 0:
            return True, "compliance_non_increasing: prev<=0, skip"

        rise = (c_curr - c_prev) / abs(c_prev)
        if rise > self.compliance_rise_tol:
            return False, (
                f"compliance_non_increasing: FAIL  "
                f"C rose by {rise:.1%}  "
                f"({c_prev:.4f} → {c_curr:.4f})"
            )
        return True, f"compliance_non_increasing: OK  (Δ={rise:+.3%})"

    def check_vol_frac(self, rho: np.ndarray,
                       void_mask: np.ndarray | None = None) -> tuple[bool, str]:
        """Volume fraction of active elements must stay near target."""
        active = rho if void_mask is None else rho[~void_mask]
        vf     = float(active.mean())
        err    = abs(vf - self.volfrac_target)
        ok     = err < self.vol_tol
        return ok, (
            f"vol_frac_within_tol: {'OK' if ok else 'FAIL'}  "
            f"actual={vf:.4f}  target={self.volfrac_target:.4f}  "
            f"err={err:.4f}  tol={self.vol_tol:.4f}"
        )

    def check_checkerboard(self, rho: np.ndarray,
                           nelx: int, nely: int) -> tuple[bool, str]:
        """
        Checkerboard detection via a 2×2 neighbourhood variance metric.

        For each 2×2 block of elements, compute the variance of the four
        densities. High mean variance indicates a checkerboard pattern where
        adjacent elements alternate 0/1.

        Score = mean variance over all 2×2 blocks, normalised to [0, 1].
        A perfect checkerboard scores 0.25 (variance of [0,0,1,1]).
        Threshold of 0.3 flags suspicious patterns.
        """
        if len(rho) != nelx * nely:
            return True, "checkerboard: SKIP (rho size mismatch)"

        img = rho.reshape((nely, nelx))

        # Build all 2×2 block patches
        patches = np.stack([
            img[:-1, :-1],
            img[:-1, 1: ],
            img[1:,  :-1],
            img[1:,  1: ],
        ], axis=-1)   # shape (nely-1, nelx-1, 4)

        block_var  = patches.var(axis=-1)        # (nely-1, nelx-1)
        score      = float(block_var.mean())
        normalised = score / 0.25                # 1.0 = worst-case checkerboard

        ok = normalised < self.checkerboard_threshold
        return ok, (
            f"no_checkerboard: {'OK' if ok else 'FAIL'}  "
            f"score={normalised:.3f}  threshold={self.checkerboard_threshold:.3f}"
        )

    # ── combined interface ─────────────────────────────────────────────────

    def check(
        self,
        rho: np.ndarray,
        nelx: int,
        nely: int,
        compliance_history: list[float],
        void_mask: np.ndarray | None = None,
        itr: int = 0,             # <-- NEW
        penal_steps: int = 0,     # <-- NEW
    ) -> tuple[bool, list[str]]:
        results = [
            self.check_compliance(compliance_history, itr, penal_steps), # <-- UPDATED
            self.check_vol_frac(rho, void_mask),
            self.check_checkerboard(rho, nelx, nely),
        ]
        passed  = all(r[0] for r in results)
        reasons = [r[1] for r in results]
        return passed, reasons
    def assert_all(
        self,
        rho: np.ndarray,
        nelx: int,
        nely: int,
        compliance_history: list[float],
        void_mask: np.ndarray | None = None,
        itr: int = 0,             # <-- NEW
        penal_steps: int = 0,     # <-- NEW
    ):
        passed, reasons = self.check(rho, nelx, nely, compliance_history, void_mask, itr, penal_steps)
        if not passed:
            failing = [r for r in reasons if "FAIL" in r]
            raise AssertionError("PhysicsGate FAILED:\n  " + "\n  ".join(failing))


# ══════════════════════════════════════════════════════════════════════════════
# 2. STEERING AGENT
# ══════════════════════════════════════════════════════════════════════════════

STEERING_SYSTEM = """\
You are an expert in structural topology optimisation using the SIMP method.

Rules:
- Only return valid JSON, no other text, no markdown fences.
- Keys you may adjust: "penal", "r_min", "volfrac_target"
- Constraints:
    penal          ∈ [1.0, 5.0]
    r_min          ∈ [0.01, 0.2]
    volfrac_target ∈ [0.1, 0.9]

Decision rules — follow these strictly in order:
1. If iteration <= 50, DO NOT change penal. The solver is still stabilizing from the initial ramp.
2. If iters_since_penal_change < 15, DO NOT change penal under any circumstances. Allow the solver to converge from the last bump.
3. If compliance_trend is "rising" and iters_since_penal_change >= 15, hold penal steady. 
4. If checkerboard_score > 0.25, increase r_min by 10% (not 20%). Do not change penal at the same time.
5. If grey_fraction > 0.25 AND compliance_trend is "falling_or_flat" AND iters_since_penal_change >= 25, raise penal by at most 0.2.
6. If l2_delta < 0.05 and compliance is stable, hold all parameters steady.
7. Never change more than one parameter per call.
8. If l2_delta is oscillating but compliance is stable, decrease 'move' limit suggestion
9: DO NOT increase penal above 3.5. If binarization is good (grey_fraction < 0.05), focus only on holding parameters steady for convergence.
Return exactly: {"penal": <float>, "r_min": <float>, "volfrac_target": <float>}
"""

STEERING_USER_TMPL = """\
Current solver state (iteration {itr}):
{state_json}

Return the updated parameters as JSON.
"""


class SteeringAgent:
    """
    Calls Claude every `every_n` iterations to read metrics and suggest
    updated solver parameters.

    Parameters returned by the LLM are validated and applied to a
    mutable `config` dict that SIMPSolver reads each iteration.

    Usage inside SIMPSolver.optimize():
        agent = SteeringAgent(config=self.config, every_n=10)
        # inside loop:
        agent.maybe_steer(itr, metrics_snapshot)
    """

    def __init__(
        self,
        config: dict,
        every_n: int = 10,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.config  = config
        self.every_n = every_n
        self.model   = model
        self.log: list[dict] = []

    def _build_state(self, itr: int, metrics_snapshot: dict) -> str:
        state = {
            "iteration":      itr,
            "current_penal":  self.config.get("penal", 3.0),
            "current_r_min":  self.config.get("r_min", 0.04),
            "current_volfrac": self.config.get("volfrac_target", 0.4),
            **metrics_snapshot,
        }
        return json.dumps(state, indent=2)

    def _call_claude(self, user_msg: str) -> str:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json",
                     "x-api-key": API_KEY,
                     "anthropic-version": "2023-06-01"},
            json={
                "model":      self.model,
                "max_tokens": 256,
                "system":     STEERING_SYSTEM,
                "messages":   [{"role": "user", "content": user_msg}],
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"].strip()

    def _parse_and_validate(self, raw: str) -> dict | None:
        """Extract JSON from LLM response, validate ranges."""
        # Strip markdown fences if the model disobeys instructions
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        start = clean.find('{')
        end   = clean.rfind('}')
        if start == -1 or end == -1:
            print(f"[SteeringAgent] No JSON object found: {raw!r}")
            return None
        clean = clean[start:end+1]
        try:
            params = json.loads(clean)
        except json.JSONDecodeError:
            print(f"[SteeringAgent] JSON parse error: {raw!r}")
            return None

        validated = {}
        validated["penal"] = float(np.clip(params.get("penal", 3.0), 1.0, self.config.get("penal_end", 5.0)))
        validated["r_min"] = float(np.clip(params.get("r_min", 0.04), 0.02, 0.08))
        validated["volfrac_target"] = float(np.clip(params.get("volfrac_target", 0.4),  0.1, 0.9))
        return validated

    def maybe_steer(self, itr: int, metrics_snapshot: dict) -> dict | None:
        """
        Call the steering agent if `itr` is a multiple of `every_n`.
        Updates self.config in place and returns the new params (or None).

        metrics_snapshot should contain keys like:
            compliance, vol_frac, l2_delta, grey_fraction, checkerboard_score
        """
        if itr % self.every_n != 0 or itr == 0:
            return None

        state_json = self._build_state(itr, metrics_snapshot)
        user_msg   = STEERING_USER_TMPL.format(itr=itr, state_json=state_json)

        try:
            raw    = self._call_claude(user_msg)
            params = self._parse_and_validate(raw)
        except Exception as e:
            print(f"[SteeringAgent] API error at iter {itr}: {e}")
            return None

        if params is None:
            return None

        # Apply to config
        self.config["penal"]          = params["penal"]
        self.config["r_min"]          = params["r_min"]
        self.config["volfrac_target"] = params["volfrac_target"]

        self.log.append({"itr": itr, "params": params, "raw": raw})
        print(
            f"[SteeringAgent @ iter {itr}] "
            f"p={params['penal']:.2f}  "
            f"r_min={params['r_min']:.3f}  "
            f"volfrac={params['volfrac_target']:.3f}"
        )
        return params


# ══════════════════════════════════════════════════════════════════════════════
# 3. CRITIC AGENT
# ══════════════════════════════════════════════════════════════════════════════

CRITIC_SYSTEM = """\
You are an expert in structural topology optimisation.
You will receive a JSON summary of a completed SIMP optimisation run.
Write a concise technical assessment (4-8 sentences) covering:
  - Whether the optimisation converged properly
  - The quality of the final topology (binarisation, grey elements)
  - Any suspicious behaviour in the compliance or density history
  - One concrete recommendation for improving the next run

Be direct. Do not use bullet points. Do not repeat the numbers back verbatim.
"""

CRITIC_USER_TMPL = """\
Completed SIMP run summary:
{summary_json}

Write your assessment.
"""


class CriticAgent:
    """
    Reads the final metrics dict and returns a natural language critique.

    Usage after SIMPSolver.optimize():
        critic = CriticAgent()
        summary = critic.critique(metrics, sanity_results)
        print(summary)
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model

    def _build_summary(
        self,
        metrics: dict,
        sanity: dict,
        rho: np.ndarray,
        nelx: int,
        nely: int,
    ) -> str:
        compliance_arr = metrics.get("compliance", [])
        l2_arr         = metrics.get("l2_delta", [])

        # Compute grey fraction
        frac_solid = float(np.mean(rho > 0.9))
        frac_void  = float(np.mean(rho < 0.1))
        frac_grey  = float(1.0 - frac_solid - frac_void)

        # Compliance trend (last 10)
        last_c = list(compliance_arr[-10:]) if len(compliance_arr) >= 10 else list(compliance_arr)
        spread = (max(last_c) - min(last_c)) / (np.mean(last_c) + 1e-30) if last_c else 0.0

        summary = {
            "total_iterations":   len(compliance_arr),
            "converged":          sanity.get("converged", "unknown"),
            "final_compliance":   float(compliance_arr[-1]) if compliance_arr else None,
            "compliance_spread_last10_pct": float(spread * 100),
            "final_l2_delta":     float(l2_arr[-1]) if l2_arr else None,
            "final_vol_frac":     float(np.mean(rho)),
            "frac_solid":         frac_solid,
            "frac_void":          frac_void,
            "frac_grey":          frac_grey,
            "physics_gate":       sanity.get("physics_gate", {}),
            "nelx":               nelx,
            "nely":               nely,
        }
        return json.dumps(summary, indent=2)

    def _call_claude(self, user_msg: str) -> str:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json",
                     "x-api-key": API_KEY,
                     "anthropic-version": "2023-06-01"},
            json={
                "model":      self.model,
                "max_tokens": 512,
                "system":     CRITIC_SYSTEM,
                "messages":   [{"role": "user", "content": user_msg}],
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"].strip()

    def critique(
        self,
        metrics: dict,
        sanity: dict,
        rho: np.ndarray,
        nelx: int,
        nely: int,
    ) -> str:
        """
        Returns a natural language critique string.

        Parameters
        ----------
        metrics : dict with keys "compliance", "l2_delta", "vol_frac" (lists)
        sanity  : dict with keys "converged" (bool), "physics_gate" (dict)
        rho     : final density array
        nelx, nely : mesh dimensions
        """
        summary_json = self._build_summary(metrics, sanity, rho, nelx, nely)
        user_msg     = CRITIC_USER_TMPL.format(summary_json=summary_json)

        try:
            return self._call_claude(user_msg)
        except Exception as e:
            return f"[CriticAgent] API error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# Integration helpers for SIMPSolver
# ══════════════════════════════════════════════════════════════════════════════

def build_metrics_snapshot(
    itr: int,
    compliance_history: list,
    l2_history: list,
    penal_history: list,          # <-- NEW ARGUMENT
    rho: np.ndarray,
    void_mask: np.ndarray | None,
    nelx: int,
    nely: int,
    domain=None,
    Lx: float = 1.0,
    Ly: float = 1.0,
) -> dict:
    """
    Build the metrics dict that SteeringAgent.maybe_steer() expects.
    Call this inside optimize() before calling maybe_steer().
    """
    # --- Spatially sort rho so checkerboard uses real neighbours ----------
    rho_spatial = rho[:nelx * nely].copy()
    if domain is not None:
        from dolfinx import mesh as dmesh
        num_local = domain.topology.index_map(domain.topology.dim).size_local
        midpoints = dmesh.compute_midpoints(
            domain, domain.topology.dim,
            np.arange(num_local, dtype=np.int32),
        )
        dx, dy = Lx / nelx, Ly / nely
        ix = np.clip(np.floor(midpoints[:, 0] / dx).astype(int), 0, nelx - 1)
        iy = np.clip(np.floor(midpoints[:, 1] / dy).astype(int), 0, nely - 1)
        rho_spatial = np.zeros(nelx * nely)
        rho_spatial[iy * nelx + ix] = rho[:num_local]

    # --- Everything below uses rho_spatial, not raw rho -------------------
    active = rho_spatial if void_mask is None else rho_spatial[~void_mask[:nelx*nely]]

    frac_solid = float(np.mean(rho_spatial > 0.9))
    frac_void  = float(np.mean(rho_spatial < 0.1))
    frac_grey  = float(1.0 - frac_solid - frac_void)

    img     = rho_spatial.reshape((nely, nelx))
    patches = np.stack([img[:-1,:-1], img[:-1,1:], img[1:,:-1], img[1:,1:]], axis=-1)
    cb_score = float(patches.var(axis=-1).mean() / 0.25)

    iters_since_penal_change = 999
    if len(penal_history) >= 2:
        # Walk backwards to find the last time it changed by more than a float tolerance
        for i in range(len(penal_history)-1, 0, -1):
            if abs(penal_history[i] - penal_history[i-1]) > 1e-3:
                iters_since_penal_change = len(penal_history) - 1 - i
                break

    last_c  = compliance_history[-5:] if len(compliance_history) >= 5 else compliance_history
    c_trend = "rising" if (len(last_c) >= 2 and last_c[-1] > last_c[0]) else "falling_or_flat"

    return {
        "compliance":         compliance_history[-1] if compliance_history else None,
        "compliance_trend":   c_trend,
        "l2_delta":           l2_history[-1] if l2_history else None,
        "vol_frac":           float(active.mean()),
        "grey_fraction":      frac_grey,
        "checkerboard_score": cb_score,
        "iters_since_penal_change": iters_since_penal_change, # <-- REPLACED PROXY
    }


# ══════════════════════════════════════════════════════════════════════════════
# Standalone smoke test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("PhysicsGate smoke test")
    print("=" * 60)

    gate = PhysicsGate(volfrac_target=0.4)
    nelx, nely = 20, 10
    n = nelx * nely

    # Good case: uniform density, flat compliance
    rho_good = np.full(n, 0.4)
    comp_good = [10.0, 9.8, 9.6, 9.5]
    passed, reasons = gate.check(rho_good, nelx, nely, comp_good)
    print(f"\n[Good case] passed={passed}")
    for r in reasons: print(f"  {r}")

    # Bad case: checkerboard pattern
    rho_cb = np.zeros(n)
    rho_cb[::2] = 1.0
    comp_bad = [9.5, 11.0]   # compliance rose 15%
    passed, reasons = gate.check(rho_cb, nelx, nely, comp_bad)
    print(f"\n[Bad case] passed={passed}")
    for r in reasons: print(f"  {r}")

    print("\n" + "=" * 60)
    print("SteeringAgent + CriticAgent require ANTHROPIC_API_KEY.")
    print("Set it and run simp.py with --steer to enable.")
    print("=" * 60)
    sys.exit(0)