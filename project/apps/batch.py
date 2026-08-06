from pathlib import Path
import sys
import os
import json
import re
import time
import traceback
from datetime import datetime
from io import StringIO

from project.parser.client import parse_problem
from project.parser.provenance import summarize_semantic_assurance
from project.topopt.controller import LEGACY_OUTPUT_ROOT, main_from_spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS_FILE = (
    PROJECT_ROOT / "experiments" / "parser_prompts.txt"
)
 
"""
To run:
/dolfinx-env/bin/python -m project.apps.batch \
    project/experiments/parser_prompts.txt
"""
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def _slug(text: str, max_len: int = 40) -> str:
    """Turn a prompt string into a safe folder-name slug."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text[:max_len] if text else "unnamed"
 
 
def _load_prompts(path: str) -> list[tuple[int, str]]:
    """
    Returns a list of (original_line_number, prompt_text) tuples.
    Skips blank lines and lines starting with #.
    Multi-line prompts are NOT supported — each physical line is one prompt.
    If you need a multi-sentence prompt, write it on one line.
    """
    prompts = []
    with open(path, "r") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            prompts.append((lineno, line))
    return prompts
 
 
# ---------------------------------------------------------------------------
# Main batch loop
# ---------------------------------------------------------------------------
 
def run_batch(prompts_file: str | os.PathLike = DEFAULT_PROMPTS_FILE):
    prompts_file = os.fspath(prompts_file)
    BATCH_DIR = str(LEGACY_OUTPUT_ROOT / "batch")
    os.makedirs(BATCH_DIR, exist_ok=True)
    import shutil
    if os.path.exists(BATCH_DIR):
        shutil.rmtree(BATCH_DIR)
    os.makedirs(BATCH_DIR)
 
    prompts = _load_prompts(prompts_file)
    total   = len(prompts)
 
    print(f"\n{'='*60}")
    print(f"  BATCH RUNNER — {total} prompts from '{prompts_file}'")
    print(f"  Output directory: {BATCH_DIR}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
 
    batch_summary = []
 
    for idx, (lineno, prompt) in enumerate(prompts, start=1):
        run_id   = f"run_{idx:03d}_{_slug(prompt)}"
        run_dir  = os.path.join(BATCH_DIR, run_id)
        log_path = os.path.join(run_dir, "run_log.txt")
        os.makedirs(run_dir, exist_ok=True)
 
        print(f"[{idx}/{total}] Starting: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"         Output: {run_dir}")
 
        run_record = {
            "run_id":         run_id,
            "prompt_line":    lineno,
            "prompt":         prompt,
            "started_at":     datetime.now().isoformat(),
            "status":         "unknown",
            "error":          None,
            "parse_ok":       False,
            "validation_passed": None,
            "final_compliance":  None,
            "final_volfrac":     None,
            "iterations":        None,
            "converged":         None,
            "design_converged":  None,
            "objective_plateau": None,
            "semantic_assurance_status": None,
            "clarification_policy": "silent_defaults",
        }
 
        t0 = time.time()
 
        try:
            # --- Parse (all defaults accepted silently) ---
            spec, defaulted_fields, field_provenance, parser_usage = parse_problem(prompt)
            run_record["parse_ok"] = True
 
            if defaulted_fields:
                defaulted_paths = [f.field_path for f in defaulted_fields]
                print(f"         Defaulted fields: {', '.join(defaulted_paths)}")
 
            # --- Run SIMP ---
            # main_from_spec is patched to accept out_dir (see SIMP_MASTER patch).
            final_field_provenance = []
            for item in field_provenance:
                record = item.model_dump()
                record["final_value"] = item.value
                record["interaction_status"] = (
                    "accepted_silently"
                    if item.source in {
                        "defaulted",
                        "inferred_from_benchmark_name",
                        "inferred_from_language",
                    }
                    else "unchanged"
                )
                final_field_provenance.append(record)

            provenance = {
                "clarification_policy": "silent_defaults",
                "defaulted_fields": [f.model_dump() for f in defaulted_fields],
                "parser_field_provenance": [
                    item.model_dump() for item in field_provenance
                ],
                "final_field_provenance": final_field_provenance,
                "clarifications_presented": [],
                "confirmed_defaults": [],
                "accepted_remaining_defaults": [
                    f.field_path for f in defaulted_fields
                ],
                "user_overrides": [],
                "opted_out": False,
                "opted_out_at_field": None,
                "final_preview_confirmed": False,
                "confirmation_received": False,
                "semantic_assurance": summarize_semantic_assurance(
                    field_provenance,
                    final_preview_confirmed=False,
                ),
                "original_prompt": prompt,
            }
            result_packet = main_from_spec(
                spec,
                parser_usage=parser_usage,
                out_dir=run_dir,
                run_provenance=provenance,
            )
 
            # --- Harvest headline numbers from result_packet ---
            if result_packet:
                val = result_packet.get("validation", {})
                fr  = result_packet.get("final_result", {})
                run_record["validation_passed"] = val.get("passed")
                run_record["final_compliance"]  = fr.get("final_compliance")
                run_record["final_volfrac"]     = fr.get("final_volume_fraction")
                run_record["iterations"]        = fr.get("iterations")
                run_record["converged"]         = fr.get("converged")
                run_record["design_converged"]  = fr.get("design_converged")
                run_record["objective_plateau"] = fr.get("objective_plateau")
                run_record["semantic_assurance_status"] = (
                    result_packet.get("interaction_provenance", {})
                    .get("semantic_assurance", {})
                    .get("status")
                )
 
            run_record["status"] = "success"
 
        except Exception as e:
            run_record["status"] = "error"
            run_record["error"]  = traceback.format_exc()
            print(f"         ERROR: {e}")
            # Write traceback into the run folder so it's findable later
            with open(os.path.join(run_dir, "error.txt"), "w") as f:
                f.write(run_record["error"])
 
        elapsed = time.time() - t0
        run_record["elapsed_seconds"] = round(elapsed, 1)
        run_record["finished_at"] = datetime.now().isoformat()
 
        status_str = (
            f"✓ passed" if run_record["validation_passed"] is True else
            f"✗ failed" if run_record["validation_passed"] is False else
            f"⚠ error"  if run_record["status"] == "error" else
            f"? unknown"
        )
        print(f"         Done in {elapsed:.0f}s — {status_str}\n")
 
        batch_summary.append(run_record)
 
        # Write incremental summary after each run so a crash doesn't
        # lose everything that ran before it.
        summary_path = os.path.join(BATCH_DIR, "batch_summary.json")
        with open(summary_path, "w") as f:
            json.dump(batch_summary, f, indent=2)
 
    # --- Final summary printout ---
    n_ok    = sum(1 for r in batch_summary if r["validation_passed"] is True)
    n_fail  = sum(1 for r in batch_summary if r["validation_passed"] is False)
    n_error = sum(1 for r in batch_summary if r["status"] == "error")
 
    print(f"\n{'='*60}")
    print(f"  BATCH COMPLETE — {total} runs")
    print(f"  Validation passed : {n_ok}")
    print(f"  Validation failed : {n_fail}")
    print(f"  Errors (crashed)  : {n_error}")
    print(f"  Summary JSON      : {summary_path}")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
 
    print("Run breakdown:")
    for r in batch_summary:
        c = f"C={r['final_compliance']:.4f}" if r['final_compliance'] else "C=n/a"
        v = f"V={r['final_volfrac']:.3f}"    if r['final_volfrac']    else "V=n/a"
        s = "✓" if r['validation_passed'] else ("✗" if r['validation_passed'] is False else "⚠")
        print(f"  {s} {r['run_id'][:50]:50s} {c}  {v}")
 
    return batch_summary
 
 
if __name__ == "__main__":
    prompts_file = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else DEFAULT_PROMPTS_FILE
    )
    if not os.path.exists(prompts_file):
        print(f"ERROR: prompts file '{prompts_file}' not found.")
        print("Create a prompts.txt file with one prompt per line.")
        sys.exit(1)
    run_batch(prompts_file)