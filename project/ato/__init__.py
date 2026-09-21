"""Agentic Topology Optimization: specification-adequacy diagnosis.

The cascade treats a run that converged correctly on a feasible stated problem
and asks whether the *stated problem* was inadequate.  Five stages:

1. convergence gate      -- refuse to diagnose a failed or infeasible run
2. anomaly detection     -- deterministic predicates on the converged field
3. hypothesis generation -- a language model proposes candidate specification
                            edits, constrained to an explicit edit grammar
4. deductive discharge   -- every candidate and every subset is applied and the
                            modified problem re-solved; survival is decided by
                            the predicates, never by the model
5. ranking + authority   -- lexicographic by admissibility tier then edit count;
                            a T3 intent-altering survivor is flagged, never applied

The model's only role is stage 3.  Tiers, boundedness and verdicts are decided
mechanically in ``grammar.py``, ``discharge.py`` and ``rank.py``.
"""
