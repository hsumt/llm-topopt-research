"""The one module the UI pages call.

It owns formulation storage and the background formulation steps (threads inside the
Streamlit process), the pre-solve check, approvals, launching an approved plan and reading
its run, and the process-flow view model. Pages import nothing else from project/ outside
project.ui.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import logging
import re
import threading
import uuid
import zipfile
from pathlib import Path

from project import ai, prompts, trace
from project.ai import MODEL
from project.drawing import geometry_preview, problem_graph
from project.examples import EXAMPLES
from project import execution
from project.execution import ARTIFACTS, run_plan
from project.lbracket import assess_spec, combine_review, reference_spec
from project.models import Review, Session
from project.ui import citations
from project.workflow import answer_questions, parse_succeeded, ready_to_try, retry_failure, start_session

FORMULATIONS = ARTIFACTS / "formulations"
APPROVALS = ARTIFACTS / "approvals"
KINDS = ("decision", "data", "unsupported")
STATES = ("working", "needs_input", "ready", "unsupported", "failed", "approved", "abandoned")
_ID = re.compile(r"[0-9a-f]{32}")

# Optional section headings for the problem text, in the order TO tools use.
OUTLINE = (
    ("Part and purpose", ""),
    ("Design space", "size, shape, areas to keep solid or keep clear"),
    ("Material and data source", ""),
    ("Supports", "where and how it is attached, and which motions that prevents"),
    ("Loads", ""),
    ("Objective and limits", ""),
    ("Manufacturing", ""),
    ("Method requirements", ""),
)
_HEADING = re.compile(r"^\s*(?:" + "|".join(re.escape(label) for label, _ in OUTLINE) + r")\b[^:\n]*:\s*",
                      re.IGNORECASE)

_log = logging.getLogger(__name__)
_lock = threading.RLock()
_threads: dict[str, threading.Thread] = {}

_BENCHMARKS = {
    "simplified_2D_holes": {
        "title": "2D L-bracket plate with five initial holes",
        "description": (
            "Equal-arm 100 mm L plate, 12 mm thick (plane stress), 60 mm corner cut, five movable 10 mm initial "
            "holes; E = 120 GPa, ν = 0.36; [0, −5000] N on the tip band; at most 75% material; p = 6 stress "
            "norm ≤ 125 MPa. Inputs of mfrto simplified_2D_holes."
        ),
        "request": "Use all documented inputs of the simplified_2D_holes benchmark.",
        "context": "Explicit selection of the five-hole reference; calibrated p6 limit, not yield.",
    },
}


@dataclass(frozen=True)
class Benchmark:
    id: str
    title: str
    description: str


@dataclass(frozen=True)
class Blocker:
    key: str
    description: str
    kind: str          # decision | data | unsupported
    origin: str        # code | model
    field: str | None


@dataclass(frozen=True)
class Precheck:
    ready: bool
    blockers: dict[str, list[Blocker]]   # always has every key in KINDS
    questions: list                      # list[models.Question]
    block_config: list[dict] | None      # [{"kind", "name", "params"}], only when ready


@dataclass(frozen=True)
class Formulation:
    fid: str
    title: str
    origin: str        # brief | benchmark | simulation
    state: str         # one of STATES
    step: str | None   # what is running while state == "working"
    error: str | None  # app-side problem; model failures are on session.failure
    created_at: str
    updated_at: str
    brief: str
    context: str | None
    session: Session | None
    approval_id: str | None


@dataclass(frozen=True)
class PlanEntry:
    fid: str
    title: str
    spec: dict
    block_config: list[dict]
    solver_inputs: dict
    budget: dict
    plan_hash: str


@dataclass(frozen=True)
class Approval:
    approval_id: str
    kind: str          # single | batch
    approver: str
    approved_at: str
    entries: list[PlanEntry]


def _heading(line: str) -> str | None:
    """The outline label a line starts with, lower-cased, or None."""
    match = _HEADING.match(line)
    return match.group(0).split("(")[0].split(":")[0].strip().lower() if match else None


def with_outline(text: str) -> str:
    """Append the outline headings that the text does not already contain."""
    present = {_heading(line) for line in text.splitlines()}
    missing = [f"{label} ({hint}):" if hint else f"{label}:"
               for label, hint in OUTLINE if label.lower() not in present]
    return "\n\n".join([text.rstrip()] * bool(text.strip()) + missing) + "\n"


def clean_brief(text: str) -> str:
    """Drop outline headings left empty; keep everything the engineer wrote."""
    kept, pending = [], None
    for line in text.splitlines():
        if _HEADING.match(line):
            pending = None if _HEADING.sub("", line, count=1).strip() else line
            if pending is None:
                kept.append(line)
        elif pending is not None:
            if line.strip():
                kept += [pending, line]
                pending = None
        else:
            kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def _title(brief: str) -> str:
    """List title before a parse: the 'Part and purpose' text, else the first written line."""
    first = purpose = section = None
    for line in brief.splitlines():
        section = _heading(line) or section
        text = _HEADING.sub("", line, count=1).strip()
        if text:
            first = first or text
            if section == "part and purpose":
                purpose = purpose or text
    text = purpose or first or ""
    return text if len(text) <= 70 else text[:67] + "…"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write(path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read(path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _dir(fid: str):
    if not _ID.fullmatch(fid or ""):
        raise ValueError("Unknown formulation.")
    return FORMULATIONS / fid


def _set_status(fid: str, **updates) -> None:
    with _lock:
        path = _dir(fid) / "status.json"
        _write(path, {**(_read(path) or {}), **updates, "updated_at": _now()})


def _save_session(fid: str, session: Session) -> None:
    _write(_dir(fid) / "session.json", session.model_dump(mode="json"))


def _load_session(fid: str) -> Session | None:
    data = _read(_dir(fid) / "session.json")
    if data is None:
        return None
    try:
        return Session.model_validate(data)
    except ValueError:
        return None


def _run(fid: str, step: str, work, round_: int) -> None:
    with trace.recording(trace.recorder_for(_dir(fid), fid, round_)):
        trace.start(step)
        try:
            session = work()
            _save_session(fid, session)
        except Exception:
            # workflow.py records model failures on the session; anything raised here is a code fault.
            _log.exception("Formulation %s: step raised", fid)
            trace.abort("code_fault")
            _set_status(fid, state=None, step=None,
                        error="The last step stopped unexpectedly. Your last saved state is kept.")
            return
        _record_present(session)
        trace.commit()
    _set_status(fid, state=None, step=None, error=None)


def _spawn(fid: str, step: str, work, round_: int = 0) -> None:
    with _lock:
        thread = _threads.get(fid)
        if thread is not None and thread.is_alive():
            raise RuntimeError("This formulation is already working on a step.")
        _set_status(fid, state="working", step=step, error=None)
        thread = threading.Thread(target=_run, args=(fid, step, work, round_), name=f"formulation-{fid}",
                                  daemon=True)
        _threads[fid] = thread
        thread.start()


@trace.guarded
def _record_present(session: Session) -> None:
    """Trace what the engineer is shown: blocking issues, and questions only while input is needed."""
    recorder = trace.current()
    if not recorder.enabled:
        return
    state, _ = _state({}, False, session)
    # A run whose model step failed wrote no merge; the session still holds the last committed one.
    own_merge = recorder.last.get("merge")
    merge = own_merge or recorder.before.get("merge")
    origins = recorder.origins if own_merge else []

    def origin(n: int, key: str) -> str:
        return origins[n] if n < len(origins) else trace.origin_of(key)

    def fate(shown: bool, rule: str) -> tuple[str, str | None]:
        if state == "failed":
            return "hidden", "ui.failed_shows_failure_only"
        return ("shown", None) if shown else ("hidden", rule)

    items = []
    for n, issue in enumerate(session.review.issues):
        shown, rule = fate(issue.blocking, "ui.blocking_only")
        items.append(trace.item("issue", issue, origin=origin(n, issue.key), fate=shown, fate_rule=rule,
                                sources=[f"{merge}#{n}"] if merge else []))
    offset = len(session.review.issues)
    for n, question in enumerate(session.review.questions):
        shown, rule = fate(state == "needs_input", "ui.questions_needs_input_only")
        items.append(trace.item("question", question, origin=origin(offset + n, question.key), fate=shown,
                                fate_rule=rule, sources=[f"{merge}#{offset + n}"] if merge else []))
    recorder.node("present", "code", "ui.api.precheck", parents=[own_merge or recorder.latest], items=items,
                  extra={"state": state, "ready": state == "ready"})


@trace.guarded
def _record_answers(session: Session, answers: dict[str, str]) -> None:
    """Trace the engineer's answers, noting where an example answer or an "Other" text was used."""
    recorder = trace.current()
    if not recorder.enabled:
        return
    present = recorder.before.get("present")
    questions = {q.key: (n, q) for n, q in enumerate(session.review.questions)}
    offset = len(session.review.issues)
    items = []
    for key, value in answers.items():
        n, question = questions.get(key, (None, None))
        detail = {"question": question.prompt if question else None,
                  "added_information": question is None and key == "additional_information"}
        if question is not None:
            detail["via_example"] = bool(question.example_answer) and value.strip() == question.example_answer.strip()
            detail["other_choice"] = (question.answer_type == "single_choice" and bool(question.options)
                                      and value not in question.options)
        items.append({"type": "answer", "origin": "engineer", "fate": "created", "key": key, "text": value,
                      "from": [f"{present}#{offset + n}"] if present and n is not None else [], "detail": detail})
    recorder.node("answer", "engineer", "ui", parents=[present], items=items)


def _answered(session: Session, answers: dict[str, str]) -> Session:
    _record_answers(session, answers)
    return answer_questions(session, answers)


def _state(status: dict, alive: bool, session: Session | None) -> tuple[str, str | None]:
    state, error = status.get("state"), status.get("error")
    if state in ("approved", "abandoned"):
        return state, error
    if state == "working":
        if alive:
            return "working", None
        return "failed", "The app restarted while a step was running. Retry the step."
    if error:
        return "failed", error
    if session is None:
        return "failed", "No specification was produced. Retry the step."
    if session.failure is not None:
        return "failed", None
    if not parse_succeeded(session):
        return "failed", "The brief has not been read successfully. Retry the step."
    if ready_to_try(session):
        return "ready", None
    if not session.review.questions and any(
            i.blocking and i.kind == "unsupported" for i in session.review.issues):
        return "unsupported", None
    return "needs_input", None


def _refuse_keys(*texts: str | None) -> None:
    # Engineer text is saved and sent to the model, so a pasted credential must never get that far.
    if any(trace.looks_like_key(text) for text in texts):
        raise ValueError("The text contains something that looks like an API key. Remove it, then try again.")


def _open_session(fid: str) -> Session:
    formulation = get(fid)
    if formulation is None:
        raise ValueError("Unknown formulation.")
    if formulation.state == "failed":
        raise ValueError("Retry the failed step first.")
    if formulation.state in ("working", "approved", "abandoned"):
        raise ValueError(f"This formulation is {formulation.state} and cannot change.")
    if formulation.session is None:
        raise ValueError("This formulation has no specification yet. Retry the step.")
    return formulation.session


def list_benchmarks() -> list[Benchmark]:
    return [Benchmark(bid, item["title"], item["description"]) for bid, item in _BENCHMARKS.items()]


def start_benchmark(benchmark_id: str) -> str:
    """Create a ready formulation from a documented benchmark; no model call."""
    item = _BENCHMARKS.get(benchmark_id)
    if item is None:
        raise ValueError("Unknown benchmark.")
    fid = uuid.uuid4().hex
    spec = reference_spec()
    # reference_spec builds parameters from a set; fix the order so the page reads the same every time.
    order = ("lx", "ly", "thickness", "cut_length", "load_patch", "hole_radius", "hole_centers")
    parameters = spec.geometry.parameters
    spec = spec.model_copy(update={"geometry": spec.geometry.model_copy(update={"parameters": {
        name: parameters[name] for name in sorted(parameters, key=lambda n: order.index(n) if n in order else len(order))}})})
    _write(_dir(fid) / "meta.json", {"fid": fid, "origin": "benchmark", "benchmark_id": benchmark_id,
                                     "brief": item["request"], "context": item["context"], "created_at": _now()})
    with trace.recording(trace.recorder_for(_dir(fid), fid)):
        trace.start("Loading the benchmark")
        trace.benchmark(spec, benchmark_id)
        session = Session(original_request=item["request"], context=item["context"], spec=spec,
                          review=combine_review(spec, Review()), parse_completed=True)
        _save_session(fid, session)
        _record_present(session)
        trace.commit()
    _set_status(fid, state=None, step=None, error=None)
    return fid


def start_formulation(brief: str, context: str | None = None) -> str:
    """Start parsing and reviewing a brief in the background; returns its id at once."""
    brief, context = clean_brief(brief), (context or "").strip() or None
    if not brief:
        raise ValueError("Describe the problem first. Empty outline headings are not counted.")
    _refuse_keys(brief, context)
    fid = uuid.uuid4().hex
    _write(_dir(fid) / "meta.json", {"fid": fid, "origin": "brief", "brief": brief,
                                     "context": context, "created_at": _now()})
    _spawn(fid, "Reading and reviewing the brief", lambda: start_session(brief, context))
    return fid


def answer(fid: str, answers: dict[str, str]) -> None:
    answers = {key: value.strip() for key, value in answers.items() if value and value.strip()}
    if not answers:
        raise ValueError("Answer at least one question.")
    _refuse_keys(*answers.values())
    session = _open_session(fid)
    _spawn(fid, "Applying your answers", lambda: _answered(session, answers), len(session.revisions))


def add_information(fid: str, text: str) -> None:
    text = text.strip()
    if not text:
        raise ValueError("Enter the information to add.")
    _refuse_keys(text)
    session = _open_session(fid)
    _spawn(fid, "Applying the added information",
           lambda: _answered(session, {"additional_information": text}), len(session.revisions))


def retry(fid: str) -> None:
    formulation = get(fid)
    if formulation is None or formulation.state != "failed":
        raise ValueError("Only a failed step can be retried.")
    session = formulation.session
    if session is None:
        brief, context = formulation.brief, formulation.context
        _spawn(fid, "Retrying: reading and reviewing the brief", lambda: start_session(brief, context))
    else:
        _spawn(fid, "Retrying the failed step", lambda: retry_failure(session), len(session.revisions))


def abandon(fid: str) -> None:
    formulation = get(fid)
    if formulation is None or formulation.state in ("working", "approved", "abandoned"):
        raise ValueError("This formulation cannot be abandoned now.")
    _set_status(fid, state="abandoned")


def get(fid: str) -> Formulation | None:
    try:
        folder = _dir(fid)
    except ValueError:
        return None
    meta = _read(folder / "meta.json")
    if meta is None or not isinstance(meta.get("brief"), str):
        return None
    with _lock:
        # Read together: a step that finishes between two separate reads would look like a crash.
        status = _read(folder / "status.json") or {}
        thread = _threads.get(fid)
        alive = thread is not None and thread.is_alive()
    session = _load_session(fid)
    state, error = _state(status, alive, session)
    title = _title(meta["brief"])
    if session is not None and parse_succeeded(session) and session.spec.name not in ("", "Unparsed request"):
        title = session.spec.name
    return Formulation(fid=fid, title=title, origin=meta.get("origin", "brief"), state=state,
                       step=status.get("step") if state == "working" else None, error=error,
                       created_at=meta.get("created_at", ""), updated_at=status.get("updated_at", ""),
                       brief=meta["brief"], context=meta.get("context"), session=session,
                       approval_id=status.get("approval_id"))


def list_formulations() -> list[Formulation]:
    items = [get(path.parent.name) for path in FORMULATIONS.glob("*/meta.json")]
    return sorted((f for f in items if f is not None), key=lambda f: f.updated_at, reverse=True)


def _block_config(config) -> list[dict]:
    """Express the 2D L-bracket solver config as a block list for display and the approval record."""
    nx, ny = config.mesh_shape
    blocks = [
        {"kind": "geometry", "name": "lshape",
         "params": {"lx_m": config.lx_m, "ly_m": config.ly_m, "thickness_m": config.thickness_m,
                    "cut_length_m": config.cut_length_m}},
        {"kind": "initial_design", "name": "holes",
         "params": {"radius_m": config.hole_radius_m, "centers_m": [list(c) for c in config.hole_centers_m]}},
        {"kind": "regions", "name": "named_edge",
         "params": {"name": "top_arm", "edge": "top edge of the vertical arm"}},
        {"kind": "regions", "name": "named_edge",
         "params": {"name": "tip_band", "edge": "upper band of the horizontal-arm tip edge",
                    "height_m": config.load_patch_m}},
        {"kind": "materials", "name": "linear_isotropic",
         "params": {"region": "domain", "youngs_modulus_pa": config.youngs_modulus_pa,
                    "poisson_ratio": config.poisson_ratio}},
        {"kind": "supports", "name": "clamp", "params": {"region": "top_arm", "components": ["x", "y"]}},
        {"kind": "initial_design", "name": "passive_solid",
         "params": {"region": "tip pad", "side_m": config.load_patch_m,
                    "note": "square under the load band that the solver keeps solid"}},
        {"kind": "loads", "name": "traction",
         "params": {"region": "tip_band", "total_force_n": list(config.force_n), "distribution": "uniform"}},
        {"kind": "objectives", "name": "compliance", "params": {"sense": "minimize"}},
        {"kind": "constraints", "name": "volume",
         "params": {"reference": "l_domain", "max_fraction": config.volume_fraction}},
    ]
    if config.stress_limit_pa is not None:
        blocks.append({"kind": "constraints", "name": "stress_pnorm",
                       "params": {"region": "l_domain", "p": config.p_norm, "limit_pa": config.stress_limit_pa}})
    blocks.append({"kind": "discretization", "name": "structured_quad",
                   "params": {"cells": [nx, ny], "element": "CG1", "model": "plane stress"}})
    return blocks


def precheck(session: Session | None) -> Precheck:
    blockers: dict[str, list[Blocker]] = {kind: [] for kind in KINDS}
    if session is None:
        return Precheck(False, blockers, [], None)
    for issue in session.review.issues:
        if issue.blocking:
            origin = "code" if issue.key.startswith("solver_") else "model"
            blockers.setdefault(issue.kind, []).append(
                Blocker(issue.key, issue.description, issue.kind, origin, issue.field))
    assessment = assess_spec(session.spec) if parse_succeeded(session) else None
    ready = ready_to_try(session) and assessment is not None and assessment.ready
    if not ready:
        return Precheck(False, blockers, list(session.review.questions), None)
    return Precheck(True, blockers, [], _block_config(assessment.config))


def _hash(data: dict) -> str:
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _plan_body(session: Session, max_iterations: int, ranks: int) -> dict:
    plan = run_plan(session, max_iterations, ranks)   # validates readiness and budget
    return {"spec": plan["spec"], "block_config": precheck(session).block_config,
            "solver_inputs": json.loads(json.dumps(plan["config"])),
            "budget": {"max_iterations": max_iterations, "mpi_ranks": ranks}}


def _run_state(status: dict) -> str | None:
    """State of the formulation's latest run (starting, running, completed, failed, interrupted), if any."""
    run_dir = Path(status["run_dir"]) if status.get("run_dir") else None
    if run_dir is None or not (run_dir / "status.json").exists():
        return None
    return execution.run_status(run_dir).get("state")


def _active_run(status: dict) -> bool:
    return _run_state(status) in ("starting", "running")


def _run_pending(status: dict) -> bool:
    """True while the current approval has not started its run: each approval starts at most one run."""
    if status.get("state") != "approved" or not status.get("approval_id"):
        return False
    return not status.get("run_dir") or status.get("run_approval_id") != status["approval_id"]


def flow_version(fid: str) -> tuple:
    """Changes whenever this formulation's process flow gains a node or a node changes state.

    Cheap to poll: no densities are read, and a run's iteration count is left out so a running
    solver does not redraw the flow every few seconds.
    """
    formulation = get(fid)
    if formulation is None:
        return ()
    status = _read(_dir(fid) / "status.json") or {}
    run_dir = Path(status["run_dir"]) if status.get("run_dir") else None
    run = execution.run_status(run_dir) if run_dir is not None and (run_dir / "status.json").exists() else {}
    return (formulation.state, formulation.approval_id, len(trace_nodes(fid)), run_dir and run_dir.name,
            run.get("state"), run.get("step"))


def run_actions(fid: str) -> dict:
    """What can happen next to an approved formulation's run.

    can_run: the current approval has not started its run yet. can_reapprove: no run is active, so a
    new approval (for example with a new budget) can replace the current one.
    """
    with _lock:
        status = _read(_dir(fid) / "status.json") or {}
        approved = status.get("state") == "approved"
        run_state = _run_state(status) if approved else None
    active = run_state in ("starting", "running")
    record = get_approval(status.get("approval_id")) if approved else None
    return {"can_run": bool(record and _run_pending(status) and not active),
            "can_reapprove": approved and not active, "active": active,
            "run_state": run_state, "pending": _run_pending(status)}


def plan_entry(fid: str, max_iterations: int = 500, ranks: int = 1) -> PlanEntry:
    """The exact plan that approving this formulation would record.

    An approved formulation can be approved again once its run is not active; the new approval
    replaces the old one for the next run.
    """
    formulation = get(fid)
    if formulation is None or formulation.state not in ("ready", "approved"):
        raise ValueError(f"{formulation.title if formulation else 'This formulation'} is not ready for approval.")
    if formulation.state == "approved" and _active_run(_read(_dir(fid) / "status.json") or {}):
        raise RuntimeError(f"{formulation.title} is running. Approve it again after the run ends.")
    body = _plan_body(formulation.session, max_iterations, ranks)
    return PlanEntry(fid=fid, title=formulation.title, plan_hash=_hash(body), **body)


def approve(items: list[tuple], approver: str) -> str:
    """Record one approval covering every (fid, max_iterations, ranks[, shown_plan_hash]) item.

    A shown hash that no longer matches the rebuilt plan rejects the whole approval,
    so what is recorded is exactly what the engineer saw. Nothing runs here: each
    approved plan starts at most one run, through launch().
    """
    approver = approver.strip()
    if not approver:
        raise ValueError("Enter your name to approve.")
    if not items:
        raise ValueError("Nothing to approve.")
    fids = [item[0] for item in items]
    if len(set(fids)) != len(fids):
        raise ValueError("A formulation appears twice in this approval.")
    with _lock:
        entries = []
        for fid, iterations, ranks, *shown in items:
            entry = plan_entry(fid, iterations, ranks)
            if shown and shown[0] != entry.plan_hash:
                raise ValueError(f"{entry.title} changed after it was shown. Review it again before approving.")
            entries.append(entry)
        approval_id = uuid.uuid4().hex
        _write(APPROVALS / f"{approval_id}.json", {
            "approval_id": approval_id, "kind": "single" if len(entries) == 1 else "batch",
            "approver": approver, "approved_at": _now(),
            "entries": [{"fid": e.fid, "title": e.title, "plan_hash": e.plan_hash,
                         "plan": {"spec": e.spec, "block_config": e.block_config,
                                  "solver_inputs": e.solver_inputs, "budget": e.budget}} for e in entries],
        })
        for fid in fids:
            _set_status(fid, state="approved", approval_id=approval_id)
    return approval_id


def approve_and_run(items: list[tuple], approver: str) -> tuple[str, dict[str, dict]]:
    """Approve exactly as approve() does, then start each approved plan's run.

    Returns the approval id and, per formulation, {"run": run folder} or {"error": why it did not
    start}. The approval stands either way; a plan that did not start can be run from its page.
    """
    approval_id = approve(items, approver)
    outcomes = {}
    for fid, *_ in items:
        try:
            outcomes[fid] = {"run": launch(fid, approval_id)}
        except (ValueError, RuntimeError, OSError) as error:
            outcomes[fid] = {"error": str(error)}
    return approval_id, outcomes


def _approval(data: dict) -> Approval | None:
    try:
        entries = [PlanEntry(fid=e["fid"], title=e["title"], plan_hash=e["plan_hash"], **e["plan"])
                   for e in data["entries"]]
        return Approval(data["approval_id"], data["kind"], data["approver"], data["approved_at"], entries)
    except (KeyError, TypeError):
        return None


def get_approval(approval_id: str) -> Approval | None:
    if not _ID.fullmatch(approval_id or ""):
        return None
    data = _read(APPROVALS / f"{approval_id}.json")
    return _approval(data) if data is not None else None


def list_approvals() -> list[Approval]:
    items = [_approval(d) for d in map(_read, APPROVALS.glob("*.json")) if d is not None]
    return sorted((a for a in items if a is not None), key=lambda a: a.approved_at, reverse=True)


def trace_nodes(fid: str) -> list[dict]:
    """Every recorded trace node of a formulation, in order (see project/trace.py)."""
    try:
        return trace.read(_dir(fid))
    except ValueError:
        return []


def preview(spec):
    """Geometry figure for a spec, or None when there is too little geometry."""
    return geometry_preview(spec)


def graph(spec) -> str:
    """Graphviz source for the problem structure."""
    # problem_graph quotes the name without escaping it.
    return problem_graph(spec.model_copy(update={"name": spec.name.replace('"', "'")}))


# ---- simulated run and process flow ------------------------------------------------------------

def simulation_available() -> bool:
    """True where the local presentation demo (project/ui/demo.py, not in the repository) exists."""
    return importlib.util.find_spec("project.ui.demo") is not None


def start_simulation() -> str:
    """Run the pre-solve pipeline on the demo's scripted model replies and approve the result."""
    from project.ui import demo
    return demo.start_simulation()


RULES = {
    "assess.merge_patch_load": "Load area and load distribution are one decision, so their two questions were merged.",
    "assess.cap_questions_4": "The code check asks at most 4 questions per round.",
    "merge.drop_solver_prefix": "The reviewer used a key reserved for code checks; code issues come only from the code check.",
    "merge.relabel_material_data": "A named material's missing constants are source data, not an engineering decision.",
    "merge.dedupe_question_key": "A question with this key is already being asked.",
    "merge.q_requires_decision_link": "Reviewer questions are asked only when they resolve a blocking decision.",
    "merge.cap_4": "At most 4 questions are shown per round; code-check questions come first.",
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
        ("Decisions get fixed question templates; at most 4 questions per round.", "project/lbracket.py#\"assess.cap_questions_4\""),
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
    "Approval record": ("The stored record of what was approved.", "project/ui/api.py#\"approver\": approver, \"approved_at\": _now()", [
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
    from project.ui import citations
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


def launch(fid: str, approval_id: str | None = None) -> str:
    """Run the solver on an approved formulation's recorded plan; returns the run folder name.

    Each approval starts at most one run, and only while the plan the current code builds is
    still exactly the approved one (same hash as the approval record). With approval_id, only
    that approval may start: a newer approval recorded in between is not run in its place.
    """
    # Held from the checks until the run is recorded, so two clicks cannot start two solvers.
    with _lock:
        formulation = get(fid)
        if formulation is None or formulation.state != "approved":
            raise ValueError("Only an approved formulation can be run.")
        if approval_id is not None and formulation.approval_id != approval_id:
            raise ValueError("A later approval replaced this one before its run started. "
                             "Run the current approval from its own page.")
        record = get_approval(formulation.approval_id)
        entry = next((e for e in record.entries if e.fid == fid), None) if record else None
        if entry is None:
            raise ValueError("The approval record for this formulation is missing.")
        status = _read(_dir(fid) / "status.json") or {}
        if _active_run(status):
            raise RuntimeError("This plan is already running.")
        if not _run_pending(status):
            raise ValueError("This approval has already started its run. Approve again to start another run.")
        iterations, ranks = entry.budget["max_iterations"], entry.budget["mpi_ranks"]
        if _hash(_plan_body(formulation.session, iterations, ranks)) != entry.plan_hash:
            raise ValueError("The solver inputs the current code builds differ from the approved plan "
                             "(the code changed after approval). Review and approve again.")
        plan = run_plan(formulation.session, iterations, ranks)
        run_dir = execution.launch_run(formulation.session, approved_hash=plan["approval_hash"],
                                       max_iterations=iterations, ranks=ranks, approver=record.approver)
        _set_status(fid, run_dir=str(run_dir), run_approval_id=record.approval_id)
    return run_dir.name


def run_info(fid: str) -> dict | None:
    """Status, convergence history, summary and final density of a formulation's latest run."""
    try:
        status = _read(_dir(fid) / "status.json") or {}
    except ValueError:
        return None
    run_dir = Path(status["run_dir"]) if status.get("run_dir") else None
    if run_dir is None or not (run_dir / "status.json").exists():
        return None
    state = execution.run_status(run_dir)
    rows = []
    try:
        for line in (run_dir / "solver" / "convergence.txt").read_text().splitlines()[1:]:
            parts = line.split()
            if len(parts) == 4:
                rows.append({"iteration": int(parts[0]), "volume_fraction": float(parts[1]),
                             "compliance": float(parts[2]), "stress_pnorm_pa": float(parts[3])})
    except (OSError, ValueError):
        pass
    summary = _read(run_dir / "solver" / "summary.json")
    frames = []
    try:
        import numpy as np
        history = np.load(run_dir / "solver" / "states" / "density_history.npz")
        frames = [{"iteration": int(i), "density": d.round(3).tolist(), "boundary_points": int(b)}
                  for i, d, b in zip(history["iterations"], history["densities"], history["boundary_points"])]
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        pass   # not written yet, or caught mid-write; the next refresh reads it
    config = _read(run_dir / "config.json") or {}
    budget = (_read(run_dir / "approval.json") or {}).get("max_iterations") or (summary or {}).get("requested_iterations")
    log_tail = None
    if state.get("state") in ("failed", "interrupted"):
        try:
            log_tail = "\n".join((run_dir / "solver.log").read_text(errors="replace").splitlines()[-25:]) or None
        except OSError:
            pass
    manifest = _read(run_dir / "paraview" / "manifest.json") or {}
    paraview = {"status": state.get("paraview"), "step": state.get("step"),
                "frames": [{"iteration": f["iteration"], "path": str(run_dir / "paraview" / f["file"])}
                           for f in manifest.get("frames", [])]}
    return {"run": run_dir.name, "state": state.get("state"), "reason": state.get("reason"), "paraview": paraview,
            "history": rows, "summary": summary, "frames": frames, "config": config,
            "budget": budget, "log_tail": log_tail, "pending": _run_pending(status)}


def element_preview(session: Session | None) -> dict | None:
    """The starting design in element form, computed by ParaLeSTO from the spec's solver config.

    None until the spec is complete enough for the code check to build a solver configuration.
    """
    if session is None or not parse_succeeded(session):
        return None
    config = assess_spec(session.spec).config
    if config is None:
        return None
    from project.solver.lbracket2d.levelset import initial_densities
    densities, _ = initial_densities(config)
    nx = config.mesh_shape[0]
    return {"density": densities.round(3).tolist(),
            "cut_cells": round(nx * config.cut_length_m / config.lx_m), "mesh": list(config.mesh_shape)}


def walkthrough_answers(fid: str) -> dict[str, str]:
    """Prepared answers when the formulation's brief is a walkthrough example."""
    formulation = get(fid)
    if formulation is None:
        return {}
    for example in EXAMPLES.values():
        if example.get("answers") and clean_brief(example["request"]) == formulation.brief:
            return dict(example["answers"])
    return {}


def flow(fid: str) -> dict:
    """The formulation's process as flowchart nodes and edges: the pre-solve steps from its trace,
    then approval, run, render and results from the approval record and run folder.

    Shapes: process (a step that does work), data (inputs and specifications), document
    (tagged review), decision (diamond), folder (stored record). Positions are grid cells.
    """
    formulation = get(fid)
    if formulation is None:
        return {"nodes": [], "edges": []}
    folder = _dir(fid)
    nodes, edges, placed = [], [], {}
    nodes_in_trace = [n for n in trace.read(folder) if not n.get("interim")]

    def add(key, shape, label, col, row, actor, detail, status="ok", sub=None):
        # Boxes are up to 150 px wide in 190 px columns and up to 92 px tall in 150 px rows,
        # so anything closer than one column and half a row would overlap.
        while any(abs(c - col) < 1 and abs(r - row) < 0.5 for c, r in placed.values()):
            row += 0.55
        placed[key] = (col, row)
        nodes.append({"id": key, "shape": shape, "label": label, "sub": sub, "col": col, "row": row,
                      "actor": actor, "status": status, "order": len(nodes),
                      "detail": {**detail, "about": _about(label)}})
        return key

    def link(a, b, label=None, dashed=False, exit=None):
        if a and b:
            edges.append({"from": a, "to": b, "label": label, "dashed": dashed, "exit": exit})

    def model_detail(node):
        model = node.get("model") or {}
        same = (node.get("actor") or {}).get("version") == trace.source_version()
        parts = prompts.PARTS.get({"critic": "review"}.get(node["step"], node["step"]), ())
        return {"actor": "model", "model": model.get("model"), "cache_hit": model.get("cache_hit"),
                "tokens": model.get("tokens"), "latency_ms": node.get("latency_ms"),
                "prompts": [{"name": name, "text": text.strip()} for name, text in parts],
                "prompt_note": None if same else "This step ran with an earlier version of the prompts; "
                                                 "the text shown is the current version.",
                "payload": _blob_text(folder, node["in"].get("payload")),
                "reply": _blob_text(folder, node["out"].get("reply")),
                "items": _items(node), "claimed_resolved": node.get("claimed_resolved")}

    prev = add("brief", "data", "Design brief", 0, 0, "engineer",
               {"text": formulation.brief, "context": formulation.context})
    spec_version, spec_ref = 0, None
    ready = assess_ref = critic_ref = None
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
    if formulation.approval_id and ready:
        record = get_approval(formulation.approval_id)
        entry = next((e for e in record.entries if e.fid == fid), None) if record else None
        _, row = placed[ready]
        a = add("approval", "process", "Approval", 9, row, "engineer",
                {"text": f"Approved by {record.approver} at {record.approved_at}." if record else None}, sub="engineer")
        link(ready, a, "YES")
        f = add("record", "folder", "Approval record", 10, row, "code",
                {"plan_hash": entry.plan_hash if entry else None,
                 "block_config": entry.block_config if entry else None,
                 "solver_inputs": entry.solver_inputs if entry else None, "budget": entry.budget if entry else None})
        link(a, f)
        info = run_info(fid)
        if info is not None and info["pending"]:
            info = None   # that run belongs to an earlier approval
        if info is None:
            run = add("run", "process", "Run", 11, row, "code",
                      {"text": "Not started yet. Run the approved plan from the formulation page."}, status="pending",
                      sub="not started")
            link(f, run, dashed=True)
        else:
            done = len(info["history"])
            state = info["state"]
            run = add("run", "process", "Run", 11, row, "code",
                      {"text": f"Run {info['run']}: {state}." + (f" {info['reason']}" if info.get("reason") else ""),
                       "metrics": (info["summary"] or {}).get("level_set") or {},
                       "history_tail": info["history"][-5:], "solver_inputs": info["config"]},
                      status="failed" if state in ("failed", "interrupted") else "ok" if state == "completed" else "pending",
                      sub=f"{state} · {done} it")
            link(f, run)
            render = None
            pv = info["paraview"]
            if pv["frames"] or (pv["status"] or {}).get("state"):
                final = pv["frames"][-1]["path"] if pv["frames"] else None
                image = ("data:image/png;base64," + base64.b64encode(Path(final).read_bytes()).decode()) if final else None
                render = add("paraview", "process", "ParaView render", 12, row, "code",
                             {"text": f"Rendered by ParaView (pvbatch) from the run's cell-data density: "
                                      f"{len(pv['frames'])} frames, one cube per element, no smoothing.",
                              "metrics": pv["status"] or {}, "image": image},
                             status="ok" if pv["frames"] else "failed", sub=f"{len(pv['frames'])} frames")
                link(run, render)
            if info["summary"]:
                last = (info["summary"] or {}).get("last_evaluated_design") or {}
                results = add("results", "data", "Results", 13 if render else 12, row, "code",
                              {"metrics": last, "summary_notes": info["summary"].get("notes"),
                               "image": _convergence_png(info)},
                              sub=f"vf {last.get('volume_fraction', 0):.3f} · C {last.get('compliance', 0):.3g}")
                link(render or run, results)
    return {"nodes": nodes, "edges": edges, "title": formulation.title, "origin": formulation.origin,
            "state": formulation.state}
