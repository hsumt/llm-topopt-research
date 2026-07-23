"""
_critic.py
"""
import json
import anthropic


SYSTEM_PROMPT = """You are a structural topology optimization critic.
You receive a compact evidence packet from a SIMP topology optimization run.

Your job:
1. Summarize convergence behavior in 2-3 sentences using the provided metrics.
2. Flag engineering anomalies using exact values when available.
3. Comment on physical plausibility for the stated problem type, but do not overclaim beyond the evidence.
4. State one specific recommendation for the next run.

Constraints:
- Do not inspect or assume access to FEM source code.
- Do not invent metrics that are not in the packet.
- Do not modify parameters.
- Do not issue an independent pass/fail verdict; refer only to the provided validation status.
- Distinguish numerical convergence from physical/manufacturing interpretability.
- No topology image content is included. Artifact filenames are not visual evidence.
- Do not claim connectivity, load-path geometry, checkerboard absence, symmetry, thin-member quality, or visible support/load consistency unless a deterministic metric for that property is explicitly included in the packet.
- Output plain text with these headings:
  Convergence Behavior
  Engineering Anomalies
  Physical Plausibility
  Recommendation
"""

def criticize(critic_input: dict):
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

   # payload = {
   #    "problem":            problem_name,
   #    "total_iterations":   metrics["iteration"],
   #    "converged":          metrics["converged"],
   #    "final_compliance":   metrics["compliance_history"][-1],
   #    "initial_compliance": metrics["compliance_history"][0],
   #    "compliance_reduction_pct": round(
   #       100 * (1 - metrics["compliance_history"][-1]
   #                / metrics["compliance_history"][0]), 1
   #    ),
   #    "final_volfrac":      metrics["volfrac_history"][-1],
   #    "final_change":       metrics["change_history"][-1],
   #    "validation_checks":  validation_result["checks"],
   
   # }
   try:
      response = client.messages.create(
         model="claude-sonnet-4-6",
         max_tokens=1200,
         system=SYSTEM_PROMPT,
         messages=[
               {"role": "user", "content": json.dumps(critic_input, indent=2)}
         ]
      )

      summary = response.content[0].text.strip()

      critic_usage = {
         "input_tokens": response.usage.input_tokens,
         "output_tokens": response.usage.output_tokens,
         "total_tokens": (
               response.usage.input_tokens
               + response.usage.output_tokens
         )
      }

      print("Critic usage:")
      print("  input tokens :", critic_usage["input_tokens"])
      print("  output tokens:", critic_usage["output_tokens"])
      print("  total tokens :", critic_usage["total_tokens"])

      return summary, critic_usage

   except Exception as e:
      return f"[CriticAgent] Unavailable({e})", {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
      }
