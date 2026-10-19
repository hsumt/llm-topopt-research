"""The process-flow view model: a formulation's trace, approval and run as flowchart nodes and edges.

Each node carries what the flow drawer shows: a short description with code citations
(ABOUT, resolved by citations.py), the recorded items and the rules that decided them,
model prompts and replies, and run results.
"""
import base64
import io
import json
from pathlib import Path

from project import prompts, trace
from project.lbracket import MAX_QUESTIONS
from project.ui import api, citations


RULES = {
    "assess.merge_patch_load": "Load area and load distribution are one decision, so their two questions were merged.",
    "assess.cap_questions_4": f"The code check asks at most {MAX_QUESTIONS} questions per round.",
    "merge.drop_solver_prefix": "The reviewer used a key reserved for code checks; code issues come only from the code check.",
    "merge.relabel_material_data": "A named material's missing constants are source data, not an engineering decision.",
    "merge.dedupe_question_key": "A question with this key is already being asked.",
    "merge.q_requires_decision_link": "Reviewer questions are asked only when they resolve a blocking decision.",
    "merge.cap_4": f"At most {MAX_QUESTIONS} questions are shown per round; code-check questions come first.",
    "ui.blocking_only": "Only blocking items are shown to the engineer.",
    "ui.questions_needs_input_only": "Questions are shown only while the formulation needs input.",
    "ui.failed_shows_failure_only": "After a failure, only the failure and the retry are shown.",
    "apply.atomic": "Edits apply all or nothing; another edit in the set was invalid.",
    "apply.path_exists": "The edit points at a field that does not exist.",
    "apply.index_bounds": "The edit points at a list position that does not exist.",
    "apply.new_key_value_maps_only": "New keys are allowed only in geometry parameters and material properties.",
    "apply.revalidate": "The edited specification failed validation.",
    "apply.empty_path": "The edit has no target path.",
}


def _rule_text(rule: str | None) -> str | None:
    if not rule:
        return None
    if rule.startswith("assess.") and "@" in rule:
        return f"Code check rule at {rule.split('@')[1]}."
    return RULES.get(rule)


def _convergence_png(info: dict) -> str | None:
    """Convergence history as a small base64 PNG for the flow drawer."""
    rows = info.get("history") or []
    if not rows:
        return None
    from matplotlib.figure import Figure
    config = info.get("config") or {}
    figure = Figure(figsize=(5.2, 4.2), dpi=96)
    axes = figure.subplots(3, 1, sharex=True)
    it = [r["iteration"] for r in rows]
    axes[0].plot(it, [r["volume_fraction"] for r in rows], color="#0b6bcb")
    if config.get("volume_fraction"):
        axes[0].axhline(config["volume_fraction"], color="#c62828", linestyle="--", linewidth=1)
    axes[0].set_ylabel("volume")
    axes[1].plot(it, [r["compliance"] for r in rows], color="#6d4fc2")
    axes[1].set_ylabel("compliance")
    axes[2].plot(it, [r["stress_pnorm_pa"] / 1e6 for r in rows], color="#b7791f")
    if config.get("stress_limit_pa"):
        axes[2].axhline(config["stress_limit_pa"] / 1e6, color="#c62828", linestyle="--", linewidth=1)
    axes[2].set_ylabel("p-norm MPa")
    axes[2].set_xlabel("iteration")
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _items(node: dict) -> list[dict]:
    return [{k: item.get(k) for k in ("type", "key", "text", "kind", "origin", "fate", "fate_rule", "rule",
                                      "why", "detail")}
            | {"rule_text": _rule_text(item.get("fate_rule") or item.get("rule")),
               "rule_cite": citations.rule(item.get("fate_rule") or item.get("rule"))} for item in node.get("items", [])]


def _blob_text(folder, ref) -> str | None:
    data = trace.read_blob(folder, ref)
    if data is None:
        return None
    return data["text"] if isinstance(data, dict) and set(data) == {"text"} else json.dumps(data, indent=2)


# Step descriptions for the flow drawer: (summary, summary anchor, [(bullet, anchor), ...]).
# Anchors are resolved by project/ui/citations.py into the code that implements each claim.
ABOUT = {
    "Design brief": ("What the engineer asked for, in their own words.", "project.ui.api:start_formulation", [
        ("Free text plus optional references and conventions; empty outline headings are dropped.", "project.ui.api:clean_brief"),
        ("Stored once and never changed; every later value must trace back to it or to an answer.", "project.ui.api:start_formulation"),
    ]),
    "Parse brief": ("A model turns the brief into a structured specification.", "project.ai:parse_problem", [
        ("Prompt parts: Parse rules and the Solver contract (hover the chips).", "project/prompts.py#\"parse\": ((\"Parse rules\""),
        ("Records only what is stated; anything unstated stays empty.", "project/prompts.py#Leave every decision missing"),
        ("The reply must match the ProblemSpec schema or the step fails.", "project.ai:_validate_response"),
    ]),
    "Parsed?": ("Code checks that the model returned a schema-valid specification.", "project.workflow:start_session", [
        ("YES: the specification is stored as version 0.", "project/workflow.py#trace.parsed(spec)"),
        ("NO: the failure is recorded and the engineer can retry.", "project.workflow:_failed"),
    ]),
    "Specification": ("The structured problem at this point in the formulation.", "project.models:ProblemSpec", [
        ("Geometry, material, supports, loads, objective and constraints.", "project.models:ProblemSpec"),
        ("Version 0 comes from the parse; later versions come from applied edits.", "project.workflow:apply_resolution"),
    ]),
    "Code check": ("Deterministic rules compare the specification with what the connected solver can run.", "project.lbracket:assess_spec", [
        ("Each rule raises a decision, data or unsupported item, tagged with its rule id.", "project/lbracket.py#def add(key, field, description"),
        (f"Decisions get fixed question templates; at most {MAX_QUESTIONS} questions per round.", "project/lbracket.py#\"assess.cap_questions_4\""),
        ("The solver configuration is built only when nothing blocks.", "project/lbracket.py#config = LBracket2DConfig.from_dict(config_values)"),
    ]),
    "Reviewer": ("A model looks for engineering decisions the code check cannot see.", "project.ai:review_problem", [
        ("Prompt parts: Review rules and Issue kinds (hover the chips).", "project/prompts.py#\"review\": ((\"Review rules\""),
        ("Sees the code's issues and how many question slots are free.", "project.workflow:_code_check"),
        ("Cannot mark anything unsupported; its items are merged, not trusted.", "project/prompts.py#Do not use \"unsupported\""),
    ]),
    "Merge": ("Code combines the code check and the reviewer into one review.", "project.lbracket:combine_review", [
        ("Code issues always survive; reviewer items with code keys are dropped.", "project/lbracket.py#\"merge.drop_solver_prefix\""),
        ("A named material's missing constants become data, not decisions.", "project/lbracket.py#\"merge.relabel_material_data\""),
        ("Reviewer questions need a blocking decision and a free slot (max 4).", "project/lbracket.py#\"merge.q_requires_decision_link\""),
    ]),
    "Review": ("The blocking items after the merge.", "project.ui.api:precheck", [
        ("Blockers are grouped as decisions, missing data or unsupported.", "project.ui.api:precheck"),
        ("Non-blocking observations are hidden from the engineer.", "project/ui/api.py#\"ui.blocking_only\""),
    ]),
    "Ready to try?": ("Can the connected solver run this exact specification?", "project.workflow:ready_to_try", [
        ("YES only when no blocking item remains and the solver configuration builds.", "project.workflow:ready_to_try"),
        ("NO: the open questions go to the engineer.", "project/ui/api.py#\"ui.questions_needs_input_only\""),
    ]),
    "Questions shown": ("What the engineer is asked this round.", "project.ui.api:_record_present", [
        ("Code-check questions come first, then reviewer questions.", "project/lbracket.py#questions = list(assessment.questions)"),
        ("No model call happens here; this is the merge's output.", "project.ui.api:_record_present"),
        ("Example answers are illustrations only and are never applied by themselves.", "project/prompts.py#example_answer texts are illustrations"),
    ]),
    "Engineer answers": ("The engineer's answers, recorded verbatim.", "project.ui.api:answer", [
        ("Notes whether an example answer or an 'Other' choice was used.", "project.ui.api:_record_answers"),
    ]),
    "Resolve answers": ("A model turns the answers into edits to the specification.", "project.ai:resolve_answers", [
        ("Prompt parts: Resolve rules and the Solver contract (hover the chips).", "project/prompts.py#\"resolve\": ((\"Resolve rules\""),
        ("Edits are proposals: paths and values that code checks next.", "project/prompts.py#proposal; code applies and checks it"),
    ]),
    "Edits valid?": ("Code applies the edits all or nothing and revalidates the specification.", "project.workflow:apply_resolution", [
        ("Paths must exist; new keys only in geometry parameters and material properties.", "project.workflow:_apply"),
        ("YES: a new specification version. NO: the edits are rejected and kept for review.", "project/workflow.py#trace.applied(session.spec, resolution, None, error)"),
    ]),
    "Edits rejected": ("The proposed edits could not be applied.", "project/workflow.py#trace.applied(session.spec, resolution, None, error)", [
        ("The engineer's answers are kept; retry asks the model again.", "project/workflow.py#\"pending_answers\": dict(answers)"),
    ]),
    "Load benchmark": ("A documented benchmark specification; no model call.", "project.lbracket:reference_spec", [
        ("Every input is stated in the benchmark.", "project.solver.lbracket2d.config:reference_config"),
    ]),
    "Approval": ("The engineer approves the exact plan.", "project.ui.api:approve", [
        ("The plan hash covers the specification, block config, solver inputs and budget.", "project.ui.api:plan_entry"),
        ("Changing anything afterwards needs a new approval.", "project/ui/api.py#changed after it was shown"),
    ]),
    "Approval record": ("The stored record of what was approved.", "project/ui/api.py#\"approver\": approver, \"approved_at\": storage.utc_now()", [
        ("Specification, block config, solver inputs, budget and plan hash.", "project/ui/api.py#\"plan\": {\"spec\": e.spec"),
    ]),
    "Run": ("The solver runs the approved plan.", "project.execution:_worker", [
        ("2D plane-stress level-set optimization (DOLFINx + ParaLeSTO), ported from mfrto.", "project.solver.lbracket2d.main:run"),
        ("The worker re-checks the approval before starting the solver.", "project/execution.py#raise _ApprovalMismatch()"),
        ("Reaching the iteration budget does not establish convergence.", "project/solver/lbracket2d/main.py#\"convergence\": \"not assessed\""),
    ]),
    "ParaView render": ("ParaView renders the run's element densities as part of the run.", "project.execution:_render_paraview", [
        ("Reads states/density.xdmf as cell data: one value per element.", "project/solver/lbracket2d/render_paraview.py#Xdmf3ReaderS(FileName"),
        ("Draws one cube per element with material; no interpolation or smoothing.", "project/solver/lbracket2d/render_paraview.py#cubes = Glyph(Input=centers"),
        ("Frames every 50 iterations and the final design, saved under paraview/.", "project/solver/lbracket2d/render_paraview.py#range(50, last, 50)"),
    ]),
    "Results": ("What the run produced.", "project/solver/lbracket2d/main.py#\"last_evaluated_design\": last_metrics", [
        ("Final volume fraction, compliance and stress p-norm.", "project/solver/lbracket2d/main.py#last_metrics = {\"iteration\""),
        ("The final design is viewable in 3D on the formulation page.", "project.ui.api:run_info"),
    ]),
}


def _about(label: str) -> dict | None:
    key = next((k for k in ABOUT if label == k or label.startswith(k)), None)
    if key is None and label.startswith("Failed"):
        return {"text": "A step failed. The reason is recorded with a controlled error code.",
                "cite": citations.cite("project.workflow:_failed"),
                "points": [{"text": "Nothing was sent to the solver.", "cite": citations.cite("project.workflow:ready_to_try")},
                           {"text": "The engineer can retry the failed step.", "cite": citations.cite("project.workflow:retry_failure")}]}
    if key is None:
        return None
    text, anchor, points = ABOUT[key]
    return {"text": text, "cite": citations.cite(anchor),
            "points": [{"text": point, "cite": citations.cite(point_anchor)} for point, point_anchor in points]}


def about_table() -> dict:
    """Every step description with its resolved code citations, keyed by node label."""
    return {label: _about(label) for label in [*ABOUT, "Failed"]}


def prompt_texts() -> dict:
    """The text of every prompt part, keyed by part name; nodes name the parts they used."""
    return {name: text.strip() for parts in prompts.PARTS.values() for name, text in parts}


def _image(path) -> str | None:
    try:
        return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()
    except OSError:
        return None


def flow(fid: str) -> dict:
    """The formulation's process as flowchart nodes and edges: the pre-solve steps from its trace,
    then approval, run, render and results from the approval record and run folder.

    Shapes: process (a step that does work), data (inputs and specifications), document
    (tagged review), decision (diamond), folder (stored record). Positions are grid cells.
    """
    formulation = api.get(fid)
    if formulation is None:
        return {"nodes": [], "edges": []}
    folder = api._dir(fid)
    nodes, edges, placed = [], [], {}
    nodes_in_trace = [n for n in trace.read(folder) if not n.get("interim")]

    def add(key, shape, label, col, row, actor, detail, status="ok", sub=None):
        # Boxes are up to 150 px wide in 190 px columns and up to 92 px tall in 150 px rows,
        # so anything closer than one column and half a row would overlap.
        while any(abs(c - col) < 1 and abs(r - row) < 0.5 for c, r in placed.values()):
            row += 0.55
        placed[key] = (col, row)
        nodes.append({"id": key, "shape": shape, "label": label, "sub": sub, "col": col, "row": row,
                      "actor": actor, "status": status, "order": len(nodes), "detail": detail})
        return key

    def link(a, b, label=None, dashed=False, exit=None):
        if a and b:
            edges.append({"from": a, "to": b, "label": label, "dashed": dashed, "exit": exit})

    def model_detail(node):
        model = node.get("model") or {}
        same = (node.get("actor") or {}).get("version") == trace.source_version()
        parts = prompts.PARTS.get(trace.MODEL_STEPS.get(node["step"]), ())
        return {"actor": "model", "model": model.get("model"), "cache_hit": model.get("cache_hit"),
                "tokens": model.get("tokens"), "latency_ms": node.get("latency_ms"),
                "prompts": [{"name": name} for name, _ in parts],
                "prompt_note": None if same else "This step ran with an earlier version of the prompts; "
                                                 "the text shown is the current version.",
                "payload": _blob_text(folder, node["in"].get("payload")),
                "reply": _blob_text(folder, node["out"].get("reply")),
                "items": _items(node), "claimed_resolved": node.get("claimed_resolved")}

    prev = add("brief", "data", "Design brief", 0, 0, "engineer",
               {"text": formulation.brief, "context": formulation.context})
    spec_version, spec_ref = 0, None
    ready = assess_ref = critic_ref = None
    base = 0
    for node in nodes_in_trace:
        step, r = node["step"], node["round"]
        base = 2 * r
        if step == "parse":
            p = add(node["id"], "process", "Parse brief", 1, base, "model", model_detail(node), sub="model")
            link(prev, p)
            d = add(node["id"] + ":ok", "decision", "Parsed?", 2, base, "code", {"text": "Did the model return a valid specification?"})
            link(p, d)
            s = add(node["id"] + ":spec", "data", "Specification v0", 3, base, "model",
                    {"spec": _blob_text(folder, node["out"].get("spec"))})
            link(d, s, "YES")
            prev = spec_ref = s
        elif step == "benchmark":
            p = add(node["id"], "process", "Load benchmark", 1, base, "code", {"text": f"Benchmark {node.get('benchmark_id')}: every input documented; no model call."})
            link(prev, p)
            s = add(node["id"] + ":spec", "data", "Specification v0", 3, base, "code", {"spec": _blob_text(folder, node["out"].get("spec"))})
            link(p, s)
            prev = spec_ref = s
        elif step == "assess":
            # The code check and the reviewer both read the spec and both feed the merge.
            assess_ref = add(node["id"], "process", "Code check", 4.5, base - 0.45, "code",
                             {"actor": "code", "items": _items(node), "config_built": node.get("config_built")}, sub="rules")
            link(spec_ref or prev, assess_ref)
        elif step == "critic":
            critic_ref = add(node["id"], "process", "Reviewer", 4.5, base + 0.45, "model", model_detail(node), sub="model")
            link(spec_ref or prev, critic_ref)
        elif step == "merge":
            m = add(node["id"], "process", "Merge", 6, base, "code",
                    {"actor": "code", "items": _items(node), "resolved_check": node.get("resolved_check")}, sub="rules")
            link(assess_ref, m)
            link(critic_ref, m)
            prev, critic_ref = m, None
        elif step == "present":
            shown = [i for i in node.get("items", []) if i.get("fate") == "shown"]
            questions = [i for i in shown if i["type"] == "question"]
            doc = add(node["id"], "document", "Review", 7, base, "code",
                      {"items": _items(node), "state": node.get("state")},
                      sub=f"{len(shown) - len(questions)} blockers · {len(questions)} questions")
            link(prev, doc)
            ready = add(node["id"] + ":ready", "decision", "Ready to try?", 8, base, "code",
                        {"text": f"State after this round: {node.get('state')}."})
            link(doc, ready)
            prev = ready
            if questions:
                from_code = sum(1 for i in questions if i.get("origin") == "code")
                q = add(node["id"] + ":questions", "document", "Questions shown", 8, base + 1, "code",
                        {"items": _items({"items": questions})},
                        sub=f"{from_code} code · {len(questions) - from_code} reviewer")
                link(ready, q, "NO")
                prev = q
        elif step == "answer":
            a = add(node["id"], "data", "Engineer answers", 7, base + 1, "engineer", {"items": _items(node)})
            link(prev, a)
            prev = a
        elif step == "resolve":
            res = add(node["id"], "process", "Resolve answers", 6, base + 1, "model", model_detail(node), sub="model")
            link(prev, res)
            prev = res
        elif step == "apply":
            ok = node["status"] == "ok"
            d = add(node["id"], "decision", "Edits valid?", 5, base + 1, "code", {"items": _items(node)})
            link(prev, d)
            if ok:
                spec_version += 1
                s = add(node["id"] + ":spec", "data", f"Specification v{spec_version}", 3, base + 2, "code",
                        {"spec": _blob_text(folder, node["out"].get("spec"))})
                link(d, s, "YES", exit="left")
                prev = spec_ref = s
            else:
                bad = add(node["id"] + ":rejected", "process", "Edits rejected", 4, base + 1, "code",
                          {"items": _items(node), "rejected": _blob_text(folder, node["out"].get("rejected"))}, status="failed")
                link(d, bad, "NO")
                prev = bad
        elif step == "failure":
            f = add(node["id"], "process", f"Failed: {node.get('failure_code')}", placed.get(prev, (1, base))[0],
                    placed.get(prev, (1, base))[1] + 0.55, "code",
                    {"text": f"Stage {node.get('stage')} failed with {node.get('failure_code')} "
                             f"({node.get('request_state')}). The engineer can retry.",
                     "reply": _blob_text(folder, (node.get("out") or {}).get("reply"))}, status="failed")
            link(prev, f, "error", dashed=True)
            prev = f
    # What is happening now: the model call in flight, the engineer's turn, or the approved run.
    active = waiting = None
    status_text = None
    call = trace.activity(folder) if formulation.state == "working" else None
    if call:
        call_base = 2 * call.get("round", 0)
        label, col, row, after = {"parse": ("Parse brief", 1, call_base, prev),
                                  "critic": ("Reviewer", 4.5, call_base + 0.45, spec_ref or prev),
                                  "resolve": ("Resolve answers", 6, call_base + 1, prev)}[call["step"]]
        parts = prompts.PARTS.get(trace.MODEL_STEPS[call["step"]], ())
        active = add("live:" + call["step"], "process", label, col, row, "model",
                     {"text": f"Model call in progress since {call.get('since')}.",
                      "prompts": [{"name": name} for name, _ in parts]},
                     status="active", sub="model · running")
        link(after, active)
        status_text = f"{label}: model call running"
    elif formulation.state == "working":
        status_text = formulation.step or "Working"
    elif formulation.state == "needs_input" and prev and prev.endswith(":questions"):
        waiting = add("next:answers", "data", "Engineer answers", 7, base + 1, "engineer",
                      {"text": "Waiting for the engineer to answer the questions shown."},
                      status="waiting", sub="waiting for you")
        link(prev, waiting, dashed=True)
        status_text = "Waiting for your answers"
    if ready and formulation.state in ("ready", "approved"):
        active, run_text = _approval_and_run(fid, formulation, placed[ready][1], ready, add, link, active)
        status_text = run_text or status_text
    return {"nodes": nodes, "edges": edges, "title": formulation.title, "origin": formulation.origin,
            "state": formulation.state, "live": api.is_live(fid), "active": active or waiting,
            "status_text": status_text}


def _approval_and_run(fid, formulation, row, ready, add, link, active):
    """Approval, approval record, run, ParaView render and results, with the step now running marked.

    Before approval and before the run starts, the steps still to come are drawn as pending.
    Returns the active node id and a status line.
    """
    record = api.get_approval(formulation.approval_id) if formulation.state == "approved" else None
    entry = next((e for e in record.entries if e.fid == fid), None) if record else None
    if formulation.state != "approved":
        a = add("approval", "process", "Approval", 9, row, "engineer",
                {"text": "Waiting for the engineer to review and approve the exact plan."},
                status="waiting", sub="waiting for you")
        link(ready, a, "YES", dashed=True)
        f = add("record", "folder", "Approval record", 10, row, "code", {"text": "Written when the plan is approved."},
                status="pending", sub="after approval")
        link(a, f, dashed=True)
        info, text, active = None, "Waiting for approval", a
    else:
        a = add("approval", "process", "Approval", 9, row, "engineer",
                {"text": f"Approved by {record.approver} at {record.approved_at}." if record else "Approved."},
                sub="engineer")
        link(ready, a, "YES")
        f = add("record", "folder", "Approval record", 10, row, "code",
                {"plan_hash": entry.plan_hash if entry else None,
                 "block_config": entry.block_config if entry else None,
                 "solver_inputs": entry.solver_inputs if entry else None, "budget": entry.budget if entry else None})
        link(a, f)
        info = api.run_info(fid, with_frames=False)
        if info is not None and info["pending"]:
            info = None   # that run belongs to an earlier approval
        text = "Approved; the run has not started" if info is None else None

    state = info["state"] if info else None
    done = len(info["history"]) if info else 0
    budget = (info or {}).get("budget") or ((entry.budget or {}).get("max_iterations") if entry else None)
    progress = f"{done}/{budget} it" if budget else f"{done} it"
    rendering = bool(info and info["paraview"]["rendering"])
    running = state in ("starting", "running")

    if info is None:
        run = add("run", "process", "Run", 11, row, "code",
                  {"text": "Starts when the approved plan is run."}, status="pending", sub="not started")
    else:
        run = add("run", "process", "Run", 11, row, "code",
                  {"text": f"Run {info['run']}: {state}." + (f" {info['reason']}" if info.get("reason") else ""),
                   "metrics": (info["summary"] or {}).get("level_set") or {},
                   "history_tail": info["history"][-5:], "solver_inputs": info["config"]},
                  status="active" if running and not rendering else "failed" if state in ("failed", "interrupted")
                  else "ok" if state == "completed" or rendering else "pending",
                  sub="starting" if state == "starting" else f"solved · {progress}" if rendering else
                  f"running · {progress}" if running else f"{state} · {progress}")
    link(f, run, dashed=info is None)
    if running and not rendering:
        active, text = run, f"Run: {'starting' if state == 'starting' else 'solving'} · {progress}"

    pv = (info or {}).get("paraview") or {"frames": [], "status": None, "rendering": False}
    pv_state = (pv["status"] or {}).get("state")
    final = pv["frames"][-1]["path"] if pv["frames"] else None
    render = add("paraview", "process", "ParaView render", 12, row, "code",
                 {"text": "Rendering the run's element densities with ParaView (pvbatch)." if rendering else
                          f"Rendered by ParaView (pvbatch) from the run's cell-data density: {len(pv['frames'])} "
                          "frames, one cube per element, no smoothing." if pv["frames"] else
                          f"No render: {(pv['status'] or {}).get('reason')}" if pv_state in ("unavailable", "failed") else
                          "Renders the run's element densities when the solver finishes.",
                  "metrics": pv["status"] or {},
                  "image": _image(final) if final else None},
                 status="active" if rendering else "ok" if pv["frames"] else
                 "failed" if pv_state in ("unavailable", "failed") else "pending",
                 sub="rendering" if rendering else f"{len(pv['frames'])} frames" if pv["frames"] else
                 pv_state or ("skipped" if state in ("failed", "interrupted") else "after the run"))
    link(run, render, dashed=not (rendering or pv["frames"] or pv_state))
    if rendering:
        active, text = render, "ParaView: rendering the result"

    summary = (info or {}).get("summary")
    if state == "completed" and summary:
        last = summary.get("last_evaluated_design") or {}
        results = add("results", "data", "Results", 13, row, "code",
                      {"metrics": last, "summary_notes": summary.get("notes"), "image": _convergence_png(info)},
                      sub=f"vf {last.get('volume_fraction', 0):.3f} · C {last.get('compliance', 0):.3g}")
        link(render, results)
        text = text or "Run completed"
    else:
        results = add("results", "data", "Results", 13, row, "code",
                      {"text": "Written when the run completes."}, status="pending",
                      sub="no results" if state in ("failed", "interrupted") else "after the run")
        link(render, results, dashed=True)
        if state in ("failed", "interrupted"):
            text = text or f"Run {state}"
    return active, text
