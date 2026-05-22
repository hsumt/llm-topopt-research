import json
import anthropic
import numpy as np


SYSTEM_PROMPT = """You are a topology optimization steering agent.
You monitor a running SIMP optimization and may adjust parameters.

You will receive a JSON metrics snapshot. You must respond with ONLY a JSON 
object no flowery prose, no explanation. You respond with this exact schema:

{
  "penal":   <float, between 1.0 and 5.0>,
  "r_min":   <float, between 0.01 and 0.2>,
  "volfrac": <float, between 0.1 and 0.9>
}

Rules:
- If compliance is not decreasing over the last 10 iterations, increase penal by 0.5
- If change is oscillating (alternating up/down), decrease r_min slightly
- If volfrac is drifting more than 0.05 from target, do not change volfrac. Other parts of the software handle this.
- If optimization looks healthy, return the current values unchanged
- Never set penal above 5.0 or below 1.5. If you're at 5.0 or at 1.5, put a lid on it.
- Never set r_min below 0.01. Try to keep it at least 0.03.
"""

def steer_code(metrics: dict, current_params: dict) -> dict:
    client = anthropic.Anthropic()
    history = metrics["compliance_history"]
    last_n = 10 
    snapshot = {
        "iteration":          metrics["iteration"],
        "current_params":     current_params,
        "compliance_last_n":  history[-last_n:],
        "compliance_trend":   "decreasing" if len(history) >= 2 
                               and history[-1] < history[-2] else "stalled",
        "change_last_n":      metrics["change_history"][-last_n:],
        "volfrac_last":       metrics["volfrac_history"][-1] if metrics["volfrac_history"] else None,
        "volfrac_target":     current_params["volfrac"],        
    }
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": json.dumps(snapshot)}
            ]
        )
        raw = response.content[0].text.strip()
        proposed = json.loads(raw)

        updated = {
            "penal":   float(np.clip(proposed["penal"],   1.5, 5.0)),
            "r_min":   float(np.clip(proposed["r_min"],   0.01, 0.2)),
            "volfrac": float(np.clip(proposed["volfrac"], 0.1, 0.9)),
        }
        return updated
    except Exception as e:
        print(f"[SteeringAgent] Failed ({e}), keeping current params.")
        return current_params