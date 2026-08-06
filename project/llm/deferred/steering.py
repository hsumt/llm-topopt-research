"""
_steering.py

re-enable needed. For novel problems. Database?
"""

import json
import anthropic
import numpy as np


SYSTEM_PROMPT = """You are a topology optimization steering agent.
You monitor a running SIMP optimization and may adjust parameters.

You will receive a JSON metrics snapshot. Respond with ONLY a JSON object —
no prose, no explanation, no markdown.

Required schema (return exactly this):
{
  "penal":   <float>,
  "r_min":   <float>,
  "volfrac": <float>
}

Rules:
1. PENAL: Only recommend increasing penal if change < 0.02 
AND grey_fraction > 0.15 AND iteration > 50. Increase by 1.0, 
clamp to 5.0. Never increase more than once per run.

2. PENAL stall: if compliance has not decreased over the last 10 iterations
   AND penal is already at its scheduled value, add an extra 0.25. Clamp
   hard to 5.0.

3. PENAL oscillation: if change values alternate up/down more than 3 times
   in the last 10, add 0.25 on top of scheduled value. Clamp to 5.0.

4. R_MIN: NEVER change r_min. Return current value unchanged always.

5. VOLFRAC: NEVER change volfrac. Return current value unchanged always.

6. If healthy and on schedule, just apply the continuation step from rule 1.
"""


def steer_code(metrics: dict, current_params: dict) -> dict:
    client = anthropic.Anthropic()
    history = metrics["compliance_history"]
    change_history = metrics["change_history"]
    last_n = 10

    snapshot = {
        "iteration":         metrics["iteration"],
        "current_params":    current_params,
        "compliance_last_n": history[-last_n:],
        "compliance_trend":  (
            "decreasing" if len(history) >= 2 and history[-1] < history[-2]
            else "stalled"
        ),
        "change_last_n":     change_history[-last_n:],
        "volfrac_last":      metrics["volfrac_history"][-1]
                             if metrics["volfrac_history"] else None,
        "volfrac_target":    current_params["volfrac"],
    }

    for attempt in range(2):   # one retry on empty/bad response
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": json.dumps(snapshot)}]
            )
            raw = response.content[0].text.strip()

            if not raw:
                raise ValueError("Empty response from steering agent")

            proposed = json.loads(raw)

            updated = {
                "penal":   float(np.clip(proposed["penal"],   1.5, 5.0)),
                "r_min":   current_params["r_min"],   # code-level guard
                "volfrac": current_params["volfrac"],  # code-level guard
            }
            return updated

        except Exception as e:
            print(f"[SteeringAgent] Attempt {attempt + 1} failed ({e})"
                  + (", retrying..." if attempt == 0 else ", keeping current params."))

    return current_params