"""
_critic.py
"""
import json
import anthropic


SYSTEM_PROMPT = """You are a structural topology optimization critic.
You receive the final metrics and validation results from a SIMP optimization run.

Your job:
1. Summarize the convergence behavior in 2-3 sentences
2. Flag any engineering anomalies (e.g. compliance plateaued early, 
   volume constraint oscillating, unusually high grey fraction)
3. Comment on whether the result is physically plausible for the problem type
4. State one specific recommendation for improving the next run

You do NOT issue pass/fail verdicts. You do NOT modify parameters.
Be concise. Use engineering language. Output plain text, no JSON.
"""

def criticize(metrics: dict, validation_result: dict, problem_name: str) -> str:
   """
   Given:
   Metrics dictionary:
      "compliance_history": [],
      "volfrac_history":    [],
      "change_history":     [],
      "l2_change_history":  [],
      "iteration":          0,
      "converged":          False,

   validation_result: output of the _physics.py

   problem_name from parser (e.g. Cantilever Beam, MBB Beam, etc.)
   """
   client = anthropic.Anthropic()

   payload = {
      "problem":            problem_name,
      "total_iterations":   metrics["iteration"],
      "converged":          metrics["converged"],
      "final_compliance":   metrics["compliance_history"][-1],
      "initial_compliance": metrics["compliance_history"][0],
      "compliance_reduction_pct": round(
         100 * (1 - metrics["compliance_history"][-1]
                  / metrics["compliance_history"][0]), 1
      ),
      "final_volfrac":      metrics["volfrac_history"][-1],
      "final_change":       metrics["change_history"][-1],
      "validation_checks":  validation_result["checks"],
   
   }
   try:
      response = client.messages.create(
         model="claude-sonnet-4-6",
         max_tokens=512,
         system=SYSTEM_PROMPT,
         messages=[
            {"role": "user", "content": json.dumps(payload)}
         ]
      )
      return response.content[0].text.strip()
   except Exception as e:
      return f"[CriticAgent] Unavailable({e})"
