"""The module the UI pages call for formulations, approvals and runs.

It owns formulation storage and the background formulation steps (threads inside the
Streamlit process), the pre-solve check, approvals, and launching an approved plan and
reading its run. The process-flow view model is project.ui.flow_model. Pages import nothing
from project/ outside project.ui.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import contextvars
import json
import logging
import math
import re
import threading
import uuid
import zipfile
from pathlib import Path

from project import execution, storage, trace
from project.ai import MODEL, PROVIDER_NAME, clarify_question   # MODEL etc. re-exported for the pages
from project.ai_errors import classify_error
from project.drawing import geometry_preview, problem_graph
from project.examples import BENCHMARK_EXAMPLE, EXAMPLES
from project.execution import ARTIFACTS, MAX_ITERATIONS, MAX_RANKS, run_plan
from project.lbracket import (CHECKED, DOMAIN, LOAD_REGION, REFERENCE_REGION, SOLVER_DEFAULTS, SUPPORT_COMPONENTS,
                              SUPPORT_REGION, TEMPLATE, assess_spec, combine_review, reference_spec, solver_inputs,
                              start_issues)
from project.models import CODE_KEY_PREFIX, ClarificationRecord, ProblemSpec, Resolution, Review, Session, Update
from project.solver.lbracket2d.config import LBracket2DConfig, reference_config
from project.ui import citations
from project.workflow import (ENGINEER_EDIT_KEY, UNPARSED_NAME, PatchRejected, answer_questions, apply_edits,
                              apply_resolution, copy_session, kept_assumptions, parse_succeeded, ready_to_try,
                              retry_failure, start_session)

FORMULATIONS = ARTIFACTS / "formulations"
APPROVALS = ARTIFACTS / "approvals"
KINDS = ("decision", "data", "unsupported")
STATES = ("working", "needs_input", "ready", "unsupported", "failed", "approved", "abandoned")
ADDED_INFORMATION_KEY = "additional_information"   # answer key for free-text information or corrections
_ABSENT = object()   # a path one side of a comparison does not have
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

def _benchmark_description() -> str:
    c = reference_config()
    mm = lambda metres: f"{metres * 1e3:g}"
    return (f"Equal-arm {mm(c.lx_m)} mm L plate, {mm(c.thickness_m)} mm thick (plane stress), "
            f"{mm(c.cut_length_m)} mm corner cut, {len(c.hole_centers_m)} movable initial holes of radius "
            f"{mm(c.hole_radius_m)} mm; E = {c.youngs_modulus_pa / 1e9:g} GPa, ν = {c.poisson_ratio:g}; "
            f"[{c.force_n[0]:g}, {c.force_n[1]:g}] N on the tip band; at most {c.volume_fraction:.0%} material; "
            f"p = {c.p_norm:g} stress norm ≤ {c.stress_limit_pa / 1e6:g} MPa. Inputs of mfrto simplified_2D_holes.")


_BENCHMARKS = {
    "simplified_2D_holes": {
        "title": "2D L-bracket plate with five initial holes",
        "description": _benchmark_description(),
        "request": "Use all documented inputs of the simplified_2D_holes benchmark.",
        "context": "Explicit selection of the five-hole reference; calibrated p6 limit, not yield.",
        "example": BENCHMARK_EXAMPLE,   # the same inputs as an editable new-problem brief (project.examples)
    },
}


@dataclass(frozen=True)
class Benchmark:
    id: str
    title: str
    description: str
    example: str | None = None   # key in EXAMPLES of its brief, to start as a new problem and change first


@dataclass(frozen=True)
class Blocker:
    key: str
    description: str
    kind: str          # decision | data | unsupported
    origin: str        # code | model
    field: str | None
    stated: str | None = None   # what the spec states at field now, as short text; None when nothing


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
    origin: str        # brief | benchmark
    state: str         # one of STATES
    step: str | None   # what is running while state == "working"
    error: str | None  # app-side problem; model failures are on session.failure
    created_at: str
    updated_at: str
    brief: str
    context: str | None
    session: Session | None
    approval_id: str | None
    parent: dict | None = None   # a copy's original: fid, title, approval_id, approver, approved_at, run


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
        storage.write_json(path, {**(_read(path) or {}), **updates, "updated_at": storage.utc_now()})


def _save_session(fid: str, session: Session) -> None:
    storage.write_json(_dir(fid) / "session.json", session.model_dump(mode="json"))


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
    for issue in _start_issues(session):   # the dry run's blockers (cached per config: no new child process)
        shown, rule = fate(True, "ui.blocking_only")
        items.append(trace.item("issue", issue, origin=trace.origin_of(issue.key), fate=shown, fate_rule=rule, sources=[]))
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
                  "added_information": question is None and key == ADDED_INFORMATION_KEY}
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
    if not session.review.questions and (any(
            i.blocking and i.kind == "unsupported" for i in session.review.issues)
            or any(i.kind == "unsupported" for i in _start_issues(session))):
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
    return [Benchmark(bid, item["title"], item["description"], item.get("example")) for bid, item in _BENCHMARKS.items()]


def benchmark_example(benchmark_id: str | None) -> str | None:
    """The EXAMPLES key of a benchmark's brief: its documented inputs to edit and start as a new problem."""
    example = (_BENCHMARKS.get(benchmark_id or "") or {}).get("example")
    return example if example in EXAMPLES else None


def start_benchmark(benchmark_id: str) -> str:
    """Create a ready formulation from a documented benchmark; no model call."""
    item = _BENCHMARKS.get(benchmark_id)
    if item is None:
        raise ValueError("Unknown benchmark.")
    fid = uuid.uuid4().hex
    spec = reference_spec()
    storage.write_json(_dir(fid) / "meta.json", {"fid": fid, "origin": "benchmark", "benchmark_id": benchmark_id,
                                     "brief": item["request"], "context": item["context"], "created_at": storage.utc_now()})
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
    storage.write_json(_dir(fid) / "meta.json", {"fid": fid, "origin": "brief", "brief": brief,
                                     "context": context, "created_at": storage.utc_now()})
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
           lambda: _answered(session, {ADDED_INFORMATION_KEY: text}), len(session.revisions))


def edit_copy(fid: str) -> str:
    """Start a new formulation from an approved one, to change it and approve it again without starting over.

    No model call: the copy starts from the approved spec, brief, context and rounds, with the reviewer's
    items merged again with a fresh code check. Its changes are checked and reviewed like any others.
    The original, its approval and its runs do not change. Returns the copy's id.
    """
    with _lock:
        formulation = get(fid)
        if formulation is None or formulation.state != "approved":
            raise ValueError("Only an approved formulation can be copied to edit.")
        if formulation.session is None:
            raise ValueError("The saved specification can no longer be read; start a new formulation.")
        status = _read(_dir(fid) / "status.json") or {}
        record = get_approval(formulation.approval_id)
        run_dir = _run_dir(status)
        parent = {"fid": fid, "title": formulation.title, "approval_id": formulation.approval_id,
                  "approver": record.approver if record else None,
                  "approved_at": record.approved_at if record else None,
                  # the run of the copied approval; None while it has not run
                  "run": run_dir.name if run_dir and not _run_pending(status) else None,
                  "rounds": len(formulation.session.revisions)}   # rounds the copy starts with
    session = formulation.session
    new = uuid.uuid4().hex
    storage.write_json(_dir(new) / "meta.json", {"fid": new, "origin": "edit", "parent": parent,
                                                 "brief": formulation.brief, "context": formulation.context,
                                                 "created_at": storage.utc_now()})
    with trace.recording(trace.recorder_for(_dir(new), new, len(session.revisions))):
        trace.start("Copying the approved formulation")
        trace.copied(session.spec, parent)
        copy = copy_session(session)
        _save_session(new, copy)
        _record_present(copy)
        trace.commit()
    _set_status(new, state=None, step=None, error=None)
    return new


def _number(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:   # an int too large for a float
        return False


def editable_values(spec: ProblemSpec) -> list[dict]:
    """The stated values the engineer can set directly in the value editor, in spec order: each
    {"field", "path", "label", "value", "unit", "kind"}, kind number, list (of numbers), text or direction
    (a load's direction, which has no unit). Values not stated yet are settled by answering or in words."""
    rows = []

    def add(path, label, item) -> None:
        if item is None or item.value is None:
            return
        value = item.value
        kind = ("number" if _number(value) else "list" if isinstance(value, list)
                else "text" if isinstance(value, str) else None)
        if kind:
            rows.append({"field": ".".join(map(str, path)), "path": list(path), "label": label,
                         "value": value, "unit": item.unit, "kind": kind})

    if spec.geometry is not None:
        for name, item in spec.geometry.parameters.items():
            add(("geometry", "parameters", name), f"Geometry: {name}", item)
    for n, material in enumerate(spec.materials):
        for key, item in material.properties.items():
            add(("materials", n, "properties", key), f"Material {material.name or n + 1}: {key}", item)
    for n, support in enumerate(spec.boundary_conditions):
        add(("boundary_conditions", n, "value"), f"Support {support.name}: value", support.value)
    for n, load in enumerate(spec.loads):
        add(("loads", n, "magnitude"), f"Load {load.name}: magnitude", load.magnitude)
        if load.direction:
            rows.append({"field": f"loads.{n}.direction", "path": ["loads", n, "direction"],
                         "label": f"Load {load.name}: direction", "value": load.direction, "unit": None,
                         "kind": "direction"})
    if spec.optimization is not None:
        for n, constraint in enumerate(spec.optimization.constraints):
            relation = f" {constraint.relation}" if constraint.relation else ""
            add(("optimization", "constraints", n, "limit"),
                f"Constraint {constraint.quantity or n + 1}{relation}: limit", constraint.limit)
    return rows


def shown_value(value, unit: str | None = None) -> str:
    """A spec or solver value as short text: a Value as "<value> <unit>", absent as "—"."""
    if isinstance(value, dict) and "value" in value and set(value) <= {"value", "unit"}:
        value, unit = value["value"], value.get("unit")
    if value is None or value is _ABSENT:
        return "—"
    text = value if isinstance(value, str) else f"{value:.6g}" if isinstance(value, float) else json.dumps(value)
    return f"{text} {unit}" if unit else text


def _parsed(row: dict, text: str):
    """The value typed into the editor for row, in the row's kind; ValueError says what is wrong."""
    text = (text or "").strip()
    if not text:
        raise ValueError(f"{row['label']}: enter a value. To remove one, describe the change in words instead.")
    if row["kind"] == "direction":
        return text
    try:
        number = int(text) if re.fullmatch(r"[+-]?\d+", text) else float(text)
    except ValueError:
        number = None
    try:
        listed = json.loads(text) if text.startswith("[") else None
    except ValueError:
        listed = None

    def numbers(item) -> bool:
        return _number(item) or (isinstance(item, list) and bool(item) and all(map(numbers, item)))

    if row["kind"] == "number" or (row["kind"] == "text" and _number(number)):
        if not _number(number):
            raise ValueError(f"{row['label']}: enter a number, for example 75 or 0.7.")
        return number
    if row["kind"] == "list" or (row["kind"] == "text" and isinstance(listed, list)):
        if not isinstance(listed, list) or not numbers(listed):
            raise ValueError(f"{row['label']}: enter a list of numbers in brackets, like the value shown.")
        return listed
    return text   # a stated value in words stays words; a number typed over it becomes a number


def edit_values(fid: str, changes: dict[str, tuple[str, str]]) -> None:
    """Set stated values directly, with no model call; the spec is then checked and reviewed again.

    changes: field (as editable_values names it) -> (value text, unit text), for the rows the engineer
    changed. The edits are applied all or nothing and must give a valid spec before the step starts.
    """
    _refuse_keys(*(text for pair in changes.values() for text in pair))
    session = _open_session(fid)
    rows = {row["field"]: row for row in editable_values(session.spec)}
    updates, notes = [], []
    for field, (text, unit) in changes.items():
        row = rows.get(field)
        if row is None:
            raise ValueError(f"{field} cannot be set here; describe the change in words instead.")
        value, unit = _parsed(row, text), (unit or "").strip() or None
        if row["kind"] == "direction":
            if unit:
                raise ValueError(f"{row['label']}: a direction has no unit.")
            if value == row["value"]:
                continue
            updates.append(Update(path=row["path"], value=value))
        else:
            if value == row["value"] and unit == row["unit"]:
                continue
            updates.append(Update(path=row["path"], value={"value": value, "unit": unit}))
        notes.append(f"{field}: {shown_value(row['value'], row['unit'])} → {shown_value(value, unit)}")
    if not updates:
        raise ValueError("No value changed. Edit a value in the table first.")
    resolution = Resolution(updates=updates)
    try:
        apply_resolution(session.spec, resolution)
    except PatchRejected as error:
        raise ValueError("These values do not form a valid specification. Check the changed rows.") from error
    note = "Values the engineer set directly: " + "; ".join(notes)
    _spawn(fid, "Applying your edits", lambda: apply_edits(session, resolution, note), len(session.revisions))


def _leaves(data, path=()):
    """(path, value) for every leaf of a spec or config dump; a Value counts as one leaf."""
    if isinstance(data, dict) and "value" in data and set(data) <= {"value", "unit"}:
        yield path, data
    elif isinstance(data, dict) and data:
        for key, value in data.items():
            yield from _leaves(value, (*path, key))
    elif isinstance(data, list) and data and all(isinstance(value, dict) for value in data):
        for n, value in enumerate(data):
            yield from _leaves(value, (*path, n))
    else:
        yield path, data


def _differences(before: dict, after: dict) -> list[dict]:
    old, new = dict(_leaves(before)), dict(_leaves(after))
    return [{"field": ".".join(map(str, path)), "approved": shown_value(old.get(path, _ABSENT)),
             "now": shown_value(new.get(path, _ABSENT))}
            for path in [*old, *(p for p in new if p not in old)] if old.get(path, _ABSENT) != new.get(path, _ABSENT)]


def changes_from_parent(fid: str) -> dict | None:
    """For a copy: what differs from the approved plan it was copied from. None for any other formulation.

    {"parent": the copy's original, "spec": [{"field", "approved", "now"}], "solver_inputs": the same for the
    solver configuration, or None while the code check cannot build one}; spec is None when the approval
    record cannot be read.
    """
    formulation = get(fid)
    if formulation is None or not formulation.parent:
        return None
    parent = formulation.parent
    record = get_approval(parent.get("approval_id") or "")
    entry = next((e for e in record.entries if e.fid == parent.get("fid")), None) if record else None
    session = formulation.session
    if entry is None or session is None:
        return {"parent": parent, "spec": None, "solver_inputs": None}
    config = assess_spec(session.spec).config if parse_succeeded(session) else None
    inputs = json.loads(json.dumps(config.to_dict())) if config is not None else None
    return {"parent": parent, "spec": _differences(entry.spec, session.spec.model_dump(mode="json")),
            "solver_inputs": _differences(entry.solver_inputs, inputs) if inputs is not None else None}


def walkthrough_answers(fid: str) -> dict[str, str]:
    """Prepared answers when the formulation's brief is a walkthrough example."""
    formulation = get(fid)
    if formulation is None:
        return {}
    for example in EXAMPLES.values():
        if example.get("answers") and clean_brief(example["request"]) == formulation.brief:
            return dict(example["answers"])
    return {}


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
    if session is not None and parse_succeeded(session) and session.spec.name:
        title = session.spec.name
    parent = meta.get("parent") if isinstance(meta.get("parent"), dict) else None
    if parent:
        title += " (copy)"
    return Formulation(fid=fid, title=title, origin=meta.get("origin", "brief"), state=state,
                       step=status.get("step") if state == "working" else None, error=error,
                       created_at=meta.get("created_at", ""), updated_at=status.get("updated_at", ""),
                       brief=meta["brief"], context=meta.get("context"), session=session,
                       approval_id=status.get("approval_id"), parent=parent)


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
         "params": {"name": SUPPORT_REGION, "edge": "top edge of the vertical arm"}},
        {"kind": "regions", "name": "named_edge",
         "params": {"name": LOAD_REGION, "edge": "upper band of the horizontal-arm tip edge",
                    "height_m": config.load_patch_m}},
        {"kind": "materials", "name": "linear_isotropic",
         "params": {"region": DOMAIN, "youngs_modulus_pa": config.youngs_modulus_pa,
                    "poisson_ratio": config.poisson_ratio}},
        {"kind": "supports", "name": "clamp",
         "params": {"region": SUPPORT_REGION, "components": list(SUPPORT_COMPONENTS)}},
        {"kind": "initial_design", "name": "passive_solid",
         "params": {"region": "tip pad", "side_m": config.load_patch_m,
                    "note": "square under the load band that the solver keeps solid"}},
        {"kind": "loads", "name": "traction",
         "params": {"region": LOAD_REGION, "total_force_n": list(config.force_n), "distribution": "uniform"}},
        {"kind": "objectives", "name": "compliance", "params": {"sense": "minimize"}},
        {"kind": "constraints", "name": "volume",
         "params": {"reference": REFERENCE_REGION, "max_fraction": config.volume_fraction}},
    ]
    if config.stress_limit_pa is not None:
        blocks.append({"kind": "constraints", "name": "stress_pnorm",
                       "params": {"region": REFERENCE_REGION, "p": config.p_norm, "limit_pa": config.stress_limit_pa}})
    blocks.append({"kind": "discretization", "name": "structured_quad",
                   "params": {"cells": [nx, ny], "element": "CG1", "model": "plane stress"}})
    return blocks


def _start_issues(session: Session) -> list:
    """The dry run's issues (lbracket.start_issues) for a session nothing else blocks: ParaLeSTO only starts from
    a built config, and an open decision would change it anyway."""
    if session.failure is not None or not parse_succeeded(session) or any(i.blocking for i in session.review.issues):
        return []
    assessment = assess_spec(session.spec)
    return start_issues(assessment.config) if assessment.ready else []


def _stated(spec, field: str | None) -> str | None:
    """What a spec (model or JSON dict) states at a field path (a.b.0.c, or a[0].c as the reviewer writes it), as
    short text; None when the path holds nothing."""
    if not field:
        return None
    node = spec.model_dump(mode="json") if hasattr(spec, "model_dump") else spec
    for part in filter(None, re.split(r"[.\[\]]+", field)):
        if isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    if node is None or node == "" or node == [] or node == {}:
        return None
    if isinstance(node, dict) and "value" in node and set(node) <= {"value", "unit"}:
        node = " ".join(str(part) for part in (node["value"], node.get("unit")) if part not in (None, ""))
    text = node if isinstance(node, str) else json.dumps(node, ensure_ascii=False, separators=(", ", ": "))
    text = " ".join(text.split())   # one line: it is shown inside list items and labels
    if not text:
        return None
    return text if len(text) <= 160 else text[:157] + "…"


def question_context(session: Session) -> dict[str, dict]:
    """Why each question in the session's review is asked, from what the system recorded (the reasoning block
    shown with it): who raised it (the code check's rule, with its source line, or the reviewer model) and, per
    issue it settles, the kind, the description, the spec field, what the spec states there now and the solver
    inputs a value there would feed."""
    assessment = assess_spec(session.spec, kept_assumptions(session))
    rules = {item.get("key"): item.get("rule") for item in assessment.trace_items if item.get("type") == "question"}
    issues = {issue.key: issue for issue in session.review.issues}
    context = {}
    for question in session.review.questions:
        code = question.key in rules   # the code check's own question, not just its key prefix
        rule = rules.get(question.key) if code else None
        linked = [issues[key] for key in question.issue_keys if key in issues]
        context[question.key] = {
            "origin": "code" if code else "model", "rule": rule, "cite": citations.rule(rule) if rule else None,
            "issues": [{"key": i.key, "kind": i.kind, "description": i.description, "field": i.field,
                        "stated": _stated(session.spec, i.field), "inputs": solver_inputs(i.field, i.key, session.spec)}
                       for i in linked]}
    return context


CLARIFY_MODES = ("explain", "reword", "ask")


def clarify(fid: str, question_key: str, mode: str, query: str = "") -> ClarificationRecord:
    """Ask the model to explain or reword one open question, or to answer the engineer's query about it. The
    reply is shown with the question and kept on the session; nothing from it reaches the spec. The call holds
    the formulation's step slot (no answer can be applied meanwhile) but leaves its state as it is."""
    if mode not in CLARIFY_MODES:
        raise ValueError("Unknown request.")
    query = (query or "").strip()
    if mode == "ask" and not query:
        raise ValueError("Type your question about it first.")
    _refuse_keys(query)
    session = _open_session(fid)
    question = next((q for q in session.review.questions if q.key == question_key), None)
    if question is None:
        raise ValueError("That question is no longer open.")
    issues = question_context(session)[question_key]["issues"]
    round_ = len(session.revisions)
    outcome: dict = {}

    def work() -> None:
        with trace.recording(trace.recorder_for(_dir(fid), fid, round_)):
            trace.start("Clarifying a question")
            try:
                reply, usage = clarify_question(question.model_dump(), issues, session.spec, mode=mode, query=query,
                                                original_request=session.original_request, context=session.context)
            except Exception as error:
                failure = classify_error(error)
                trace.failure("clarify", failure)
                outcome["record"] = ClarificationRecord(round=round_, question_key=question_key, mode=mode, query=query,
                                                        failure=f"{failure.message} {failure.next_step}",
                                                        at=storage.utc_now())
            else:
                trace.clarified(question_key, mode, query, reply)
                outcome["usage"] = usage
                outcome["record"] = ClarificationRecord(round=round_, question_key=question_key, mode=mode, query=query,
                                                        explanation=reply.explanation, rewording=reply.rewording,
                                                        at=storage.utc_now())
            trace.commit()
            with _lock:   # keep whatever else the session gained meanwhile
                current = _load_session(fid) or session
                _save_session(fid, current.model_copy(update={
                    "clarifications": [*current.clarifications, outcome["record"]],
                    "usage": [*current.usage, *([outcome["usage"]] if "usage" in outcome else [])]}))

    with _lock:
        thread = _threads.get(fid)
        if thread is not None and thread.is_alive():
            raise RuntimeError("This formulation is working on a step. Ask again when it finishes.")
        # The caller's context goes along (ai.scripted for offline runs); the trace recording is the thread's own.
        thread = threading.Thread(target=contextvars.copy_context().run, args=(work,), name=f"clarify-{fid}",
                                  daemon=True)
        _threads[fid] = thread
        thread.start()
    thread.join()
    if "record" not in outcome:
        raise RuntimeError("The request stopped unexpectedly. Try again.")
    return outcome["record"]


def precheck(session: Session | None) -> Precheck:
    blockers: dict[str, list[Blocker]] = {kind: [] for kind in KINDS}
    if session is None:
        return Precheck(False, blockers, [], None)
    for issue in session.review.issues:
        if issue.blocking:
            origin = trace.origin_of(issue.key)
            blockers.setdefault(issue.kind, []).append(
                Blocker(issue.key, issue.description, issue.kind, origin, issue.field, _stated(session.spec, issue.field)))
    for issue in _start_issues(session):
        blockers[issue.kind].append(Blocker(issue.key, issue.description, issue.kind, trace.origin_of(issue.key),
                                            issue.field, _stated(session.spec, issue.field)))
    assessment = assess_spec(session.spec) if parse_succeeded(session) else None
    ready = ready_to_try(session) and assessment is not None and assessment.ready
    if not ready:
        return Precheck(False, blockers, list(session.review.questions), None)
    return Precheck(True, blockers, [], _block_config(assessment.config))


def _plan_body(session: Session, max_iterations: int, ranks: int) -> dict:
    plan = run_plan(session, max_iterations, ranks)   # validates readiness and budget
    return {"spec": plan["spec"], "block_config": precheck(session).block_config,
            "solver_inputs": json.loads(json.dumps(plan["config"])),
            "budget": {"max_iterations": max_iterations, "mpi_ranks": ranks}}


def _run_dir(status: dict) -> Path | None:
    """The formulation's latest run folder, once the run has recorded its status."""
    run_dir = Path(status["run_dir"]) if status.get("run_dir") else None
    return run_dir if run_dir is not None and (run_dir / "status.json").exists() else None


def _run_state(status: dict) -> str | None:
    """State of the formulation's latest run (starting, running, completed, failed, interrupted), if any."""
    run_dir = _run_dir(status)
    return execution.run_status(run_dir).get("state") if run_dir else None


def _active_run(status: dict) -> bool:
    return _run_state(status) in ("starting", "running")


def _run_pending(status: dict) -> bool:
    """True while the current approval has not started its run: each approval starts at most one run."""
    if status.get("state") != "approved" or not status.get("approval_id"):
        return False
    return not status.get("run_dir") or status.get("run_approval_id") != status["approval_id"]


def is_live(fid: str) -> bool:
    """True while something is happening to this formulation: a formulation step or its solver run."""
    formulation = get(fid)
    return formulation is not None and (formulation.state == "working" or run_actions(fid)["active"])


def run_actions(fid: str) -> dict:
    """What can happen next to an approved formulation's run.

    can_run: the current approval has not started its run yet. can_reapprove: no run is active, so a
    new approval (for example with a new budget) can replace the current one.
    """
    with _lock:
        status = _read(_dir(fid) / "status.json") or {}
        approved = status.get("state") == "approved" and _load_session(fid) is not None
        run_state = _run_state(status) if approved else None
    active = run_state in ("starting", "running")
    record = get_approval(status.get("approval_id")) if approved else None
    return {"can_run": bool(record and _run_pending(status) and not active),
            "can_reapprove": approved and not active, "active": active,
            "run_state": run_state, "pending": _run_pending(status)}


def run_info(fid: str, with_frames: bool = True) -> dict | None:
    """Status, convergence history, summary and density frames of a formulation's latest run.

    with_frames=False skips loading the density snapshots, for callers that poll during a run.
    """
    try:
        status = _read(_dir(fid) / "status.json") or {}
    except ValueError:
        return None
    run_dir = _run_dir(status)
    if run_dir is None:
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
    if with_frames:
        try:
            import numpy as np
            history = np.load(run_dir / "solver" / "states" / "density_history.npz")
            frames = [{"iteration": int(i), "density": d.round(3).tolist(), "boundary_points": int(b)}
                      for i, d, b in zip(history["iterations"], history["densities"], history["boundary_points"])]
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            pass   # not written yet, or caught mid-write; the next refresh reads it
    config = _read(run_dir / "config.json") or {}
    try:
        cut_cells = LBracket2DConfig.from_dict(config).cut_cells
    except (ValueError, TypeError):
        cut_cells = 0
    budget = (_read(run_dir / "approval.json") or {}).get("max_iterations") or (summary or {}).get("requested_iterations")
    log_tail = None
    if state.get("state") in ("failed", "interrupted"):
        try:
            log_tail = "\n".join((run_dir / "solver.log").read_text(errors="replace").splitlines()[-25:]) or None
        except OSError:
            pass
    manifest = _read(run_dir / "paraview" / "manifest.json") or {}
    running = state.get("state") in ("starting", "running")
    # Each frame: view name -> PNG. A render from before the standard views holds one "default" image.
    paraview = {"status": state.get("paraview"), "rendering": running and state.get("step") == execution.RENDERING_STEP,
                "frames": [{"iteration": f["iteration"],
                            "views": {name: str(run_dir / "paraview" / file)
                                      for name, file in (f.get("views") or {"default": f["file"]}).items()}}
                           for f in manifest.get("frames", [])]}
    return {"run": run_dir.name, "state": state.get("state"), "reason": state.get("reason"), "paraview": paraview,
            "history": rows, "summary": summary, "frames": frames, "config": config, "cut_cells": cut_cells,
            "budget": budget, "log_tail": log_tail, "pending": _run_pending(status)}


def plan_entry(fid: str, max_iterations: int, ranks: int) -> PlanEntry:
    """The exact plan that approving this formulation would record.

    An approved formulation can be approved again once its run is not active; the new approval
    replaces the old one for the next run.
    """
    formulation = get(fid)
    if formulation is None or formulation.state not in ("ready", "approved"):
        raise ValueError(f"{formulation.title if formulation else 'This formulation'} is not ready for approval.")
    if formulation.session is None:
        raise ValueError("The saved specification can no longer be read; start a new formulation.")
    if formulation.state == "approved" and _active_run(_read(_dir(fid) / "status.json") or {}):
        raise RuntimeError(f"{formulation.title} is running. Approve it again after the run ends.")
    body = _plan_body(formulation.session, max_iterations, ranks)
    return PlanEntry(fid=fid, title=formulation.title, plan_hash=storage.digest(body), **body)


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
        storage.write_json(APPROVALS / f"{approval_id}.json", {
            "approval_id": approval_id, "kind": "single" if len(entries) == 1 else "batch",
            "approver": approver, "approved_at": storage.utc_now(),
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
        if formulation.session is None:
            raise ValueError("The saved specification can no longer be read; start a new formulation.")
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
        if storage.digest(_plan_body(formulation.session, iterations, ranks)) != entry.plan_hash:
            raise ValueError("The solver inputs the current code builds differ from the approved plan "
                             "(the code changed after approval). Review and approve again.")
        run_dir = execution.launch_run(formulation.session, max_iterations=iterations, ranks=ranks,
                                       approver=record.approver)
        _set_status(fid, run_dir=str(run_dir), run_approval_id=record.approval_id)
    return run_dir.name


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


def preview(spec, dark: bool = False):
    """Geometry figure for a spec, or None when there is too little geometry."""
    return geometry_preview(spec, dark=dark)


def graph(spec, dark: bool = False) -> str:
    """Graphviz source for the problem structure."""
    return problem_graph(spec, dark=dark)


def spec_roles(spec, kept: frozenset[str] = frozenset()) -> dict:
    """What the connected solver makes of each part of a spec (ProblemSpec or its JSON dict).

    sources: spec path -> {"field": config field, "value": SI value in the built config, or None
    until the config builds}; checked: spec path -> what the solver requires there (lbracket.CHECKED,
    "*" for any constraint); issues: the code check's issues (key, field, kind, ...), with the dry run's
    (lbracket.start_issues) once the config builds; defaults: the config fields no spec
    sets, with their values; connected: the spec targets this solver's template or a config was built.
    """
    try:
        spec = ProblemSpec.model_validate(spec)
    except ValueError:   # e.g. a record written by an older spec model: nothing can be attributed
        return {"sources": {}, "checked": {}, "issues": [], "defaults": [], "connected": False}
    assessment = assess_spec(spec, kept)
    config = assessment.config
    values = config.to_dict() if config else {f.name: f.default for f in fields(LBracket2DConfig)}
    issues = assessment.issues + start_issues(config if assessment.ready else None)
    return {
        "sources": {path: {"field": name, "value": values[name] if config else None}
                    for name, path in assessment.sources.items()},
        "checked": dict(CHECKED),
        "issues": [{"key": i.key, "field": i.field, "kind": i.kind, "blocking": i.blocking, "description": i.description}
                   for i in issues],
        "defaults": [{"field": name, "label": label, "value": values[name]} for name, label in SOLVER_DEFAULTS.items()],
        "connected": config is not None or (spec.geometry is not None and spec.geometry.template == TEMPLATE),
    }


def element_preview(session: Session | None) -> dict | None:
    """The starting design in element form, computed by ParaLeSTO from the spec's solver config.

    None until the spec is complete enough for the code check to build a solver configuration.
    """
    if session is None or not parse_succeeded(session):
        return None
    config = assess_spec(session.spec).config
    if config is None:
        return None
    from project.solver.lbracket2d.dryrun import dry_run   # shared with the code check's dry run (lbracket.start_issues)
    run = dry_run(config)
    if run.densities is None:   # the config built, but ParaLeSTO could not start from it
        return {"error": run.failure, "config": config.to_dict()}
    return {"density": run.densities, "cut_cells": config.cut_cells, "mesh": list(config.mesh_shape), "config": config.to_dict()}
