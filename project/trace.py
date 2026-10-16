"""Append-only trace of how a formulation's questions were derived.

A formulation folder gets trace.jsonl (one JSON node per line) and blobs/
(content-addressed JSON: specs, model payloads, raw replies, rejected edits).
A node records one step (parse, assess, critic, merge, present, answer, resolve,
apply, failure, run.*), its actor, the nodes it used, and every issue, question,
answer or update it created, kept, dropped or applied, with the rule that decided.

Recording only observes. It never raises into the workflow and never changes what
the workflow returns. Code records through current(), which outside a recording()
block is a recorder that does nothing, so the CLI and the UI's own checks are
unaffected. Never stored: the API key (key-like strings are redacted), SDK
exception text, environment values.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import functools
import hashlib
import json
import logging
import re
import threading
import uuid
from pathlib import Path

VERSION = 1
TEXT_LIMIT = 300                  # characters per text field on an item
LINE_LIMIT = 64 * 1024            # bytes per node line; larger item lists spill into a blob
BLOB_LIMIT = 1024 * 1024          # bytes per blob; larger ones become a stub
TRACE_LIMIT = 5 * 1024 * 1024     # bytes of trace.jsonl per formulation
BLOBS_LIMIT = 20 * 1024 * 1024    # bytes of blobs per formulation
_KEY = re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}")              # redaction: errs towards over-redacting
_KEY_START = re.compile(r"(?<![A-Za-z0-9])sk-ant", re.IGNORECASE)   # intake: any key-like start
_TEXT_FIELDS = ("text", "why", "example_answer", "basis")
_OMITTED = ("output_schema", "output_spec_schema", "prior_rounds")   # rebuilt from the source version and revisions
_SOURCES = ("lbracket.py", "prompts.py", "models.py")
_MODEL_STEPS = {"parse": "parse", "critic": "review", "resolve": "resolve"}   # trace step -> ai.py call step

_log = logging.getLogger(__name__)
_lock = threading.RLock()   # re-entered when a node's items spill into a blob
_version: str | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def source_version() -> str:
    """Hash of the code that decides questions, so a trace names the rules that applied."""
    global _version
    if _version is None:
        digest = hashlib.sha256()
        for name in _SOURCES:
            digest.update((Path(__file__).parent / name).read_bytes())
        _version = "src@" + digest.hexdigest()[:12]
    return _version


def looks_like_key(text: str | None) -> bool:
    return bool(text) and bool(_KEY_START.search(text))


def _encode(data: dict) -> tuple[str, bool]:
    text = json.dumps(data, sort_keys=True, default=str)
    return (_KEY.sub("[redacted]", text), True) if _KEY.search(text) else (text, False)


def read(folder: Path) -> list[dict]:
    """Every well-formed node in order; damaged lines are skipped."""
    nodes = []
    try:
        lines = (Path(folder) / "trace.jsonl").read_text(errors="replace").splitlines()
    except OSError:
        return nodes
    for line in lines:
        try:
            node = json.loads(line)
        except ValueError:
            continue
        if isinstance(node, dict) and isinstance(node.get("seq"), int) and isinstance(node.get("step"), str):
            nodes.append(node)
    return nodes


def read_blob(folder: Path, ref: str):
    """The content behind a "blob:<sha>" reference, or None for stubs and missing blobs."""
    if not isinstance(ref, str) or not re.fullmatch(r"blob:[0-9a-f]{64}", ref):
        return None
    try:
        return json.loads((Path(folder) / "blobs" / f"{ref[5:]}.json").read_text())
    except (OSError, ValueError):
        return None


def guarded(function):
    """Recording helpers log and swallow their own errors; they never raise into the workflow."""
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception:
            _log.warning("trace: %s failed", function.__name__, exc_info=True)
            return None
    return wrapper


class NullRecorder:
    """Records nothing; the default outside recording()."""

    enabled = False


class Recorder:
    enabled = True

    def __init__(self, folder: Path, fid: str, round: int = 0, run: str | None = None):
        self.folder, self.fid, self.round = Path(folder), fid, round
        self.run = run or uuid.uuid4().hex[:8]
        self.last: dict[str, str] = {}       # step -> latest node id written by this run
        self.before: dict[str, str] = {}     # step -> latest node id of earlier committed runs
        self.spec_node: str | None = None    # latest node that produced the current spec
        self.latest: str | None = None
        self.call: dict | None = None        # pending model call, taken by the next model node
        self.claimed: list[str] | None = None
        self.origins: list[str] = []         # origins of the latest merge's kept items, in merged order
        self._counts: dict[str, int] = {}
        self._seq = 0
        self._trace_bytes = self._blob_bytes = 0
        self._newline = False                # the file ends in a torn line
        self._scan()

    def _scan(self) -> None:
        """Pick up where earlier runs left off, counting only runs that committed."""
        runs: dict[str, dict] = {}
        for node in read(self.folder):
            self._seq = max(self._seq, node["seq"])
            run = runs.setdefault(node.get("run"), {"last": {}, "spec": None, "claimed": None})
            run["last"][node["step"]] = node.get("id")
            if node["step"] in ("parse", "benchmark") or (node["step"] == "apply" and node.get("status") == "ok"):
                run["spec"] = node.get("id")
            if node["step"] == "resolve":
                run["claimed"] = node.get("claimed_resolved")
            elif node["step"] == "merge" and not node.get("interim"):
                run["claimed"] = []          # a final merge consumed the claim
            if node["step"] == "run.commit":
                self.before.update(run["last"])
                self.spec_node = run["spec"] or self.spec_node
                if run["claimed"] is not None:
                    self.claimed = run["claimed"] or None
        path = self.folder / "trace.jsonl"
        try:
            self._trace_bytes = path.stat().st_size
            with path.open("rb") as file:
                file.seek(-1, 2)
                self._newline = file.read(1) != b"\n"
        except OSError:
            pass
        try:
            self._blob_bytes = sum(p.stat().st_size for p in (self.folder / "blobs").glob("*.json"))
        except OSError:
            pass

    def blob(self, data) -> str:
        """Store data once by content; returns "blob:<sha>" or "stub:<sha>:<bytes>" when over a limit."""
        try:
            text, _ = _encode(data)
            sha = hashlib.sha256(text.encode()).hexdigest()
            path = self.folder / "blobs" / f"{sha}.json"
            if path.exists():
                return f"blob:{sha}"
            size = len(text.encode())
            with _lock:
                if size > BLOB_LIMIT or self._blob_bytes + size > BLOBS_LIMIT:
                    return f"stub:{sha}:{size}"
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(f".{sha}.{uuid.uuid4().hex}.tmp")
                try:
                    temporary.write_text(text)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
                self._blob_bytes += size
            return f"blob:{sha}"
        except Exception:
            _log.warning("trace: could not store a blob", exc_info=True)
            return "stub:error"

    def small(self, value):
        """A value small enough to sit in a node, or a blob reference to it."""
        text = json.dumps(value, default=str)
        return value if len(text) <= TEXT_LIMIT else self.blob(value)

    def node(self, step: str, actor: str, name: str, *, parents=(), status: str = "ok",
             failure_code: str | None = None, interim: bool = False, inputs: dict | None = None,
             outputs: dict | None = None, items=(), model: dict | None = None,
             latency_ms: int | None = None, extra: dict | None = None) -> str | None:
        try:
            return self._node(step, actor, name, parents, status, failure_code, interim, inputs,
                              outputs, items, model, latency_ms, extra)
        except Exception:
            _log.warning("trace: could not record a %s node", step, exc_info=True)
            return None

    def _node(self, step, actor, name, parents, status, failure_code, interim, inputs, outputs,
              items, model, latency_ms, extra):
        count = self._counts.get(step, 0)
        self._counts[step] = count + 1
        node_id = f"r{self.round}.{step}.{self.run}.{count}"
        node = {
            "v": VERSION, "fid": self.fid, "run": self.run, "round": self.round, "id": node_id,
            "step": step, "interim": interim, "at": _now(), "latency_ms": latency_ms,
            "actor": {"kind": actor, "name": name, "version": source_version()},
            "parents": [p for p in parents if p], "status": status, "failure_code": failure_code,
            "in": inputs or {}, "out": outputs or {}, "model": model, "items": [],
        }
        for index, item in enumerate(items):
            item = {**item, "id": f"{node_id}#{index}"}
            for field in _TEXT_FIELDS:
                if isinstance(item.get(field), str) and len(item[field]) > TEXT_LIMIT:
                    item[field] = item[field][:TEXT_LIMIT]
                    item["truncated"] = True
            node["items"].append(item)
        node.update(extra or {})
        node["seq"] = 0
        if len(_encode(node)[0].encode()) > LINE_LIMIT - 64:
            node["items_blob"], node["items"] = self.blob(node["items"]), []
        with _lock:
            node["seq"] = self._seq + 1
            line, redacted = _encode(node)
            if redacted:
                node["redacted"] = True
                line, _ = _encode(node)
            text = ("\n" if self._newline else "") + line + "\n"
            if self._trace_bytes + len(text.encode()) > TRACE_LIMIT:
                _log.warning("trace: %s is full; %s not recorded", self.fid, node_id)
                return None
            self.folder.mkdir(parents=True, exist_ok=True)
            try:
                with (self.folder / "trace.jsonl").open("a") as file:
                    file.write(text)
            except OSError:
                self._newline = True         # a partial write may have left a torn line
                raise
            self._newline = False
            self._seq += 1
            self._trace_bytes += len(text.encode())
        self.last[step] = self.latest = node_id
        return node_id

    def take_call(self, step: str) -> dict | None:
        call, self.call = self.call, None
        return call if call and call.get("step") == _MODEL_STEPS.get(step) else None

    def model_record(self, call: dict | None) -> dict:
        """Node fields for a model call: payload and reply blobs, model info, latency."""
        if call is None:
            return {}
        payload = call.get("payload") or {}
        inputs = {"payload": self.blob({k: v for k, v in payload.items() if k not in _OMITTED}),
                  "omitted": [k for k in _OMITTED if k in payload]}
        outputs = {"reply": self.blob({"text": call["reply"]})} if call.get("reply") is not None else {}
        model = {k: call.get(k) for k in ("model", "cache_key", "cache_hit", "tokens", "stop_reason")}
        return {"inputs": inputs, "outputs": outputs, "model": model, "latency_ms": call.get("latency_ms")}


_NULL = NullRecorder()
_current: ContextVar = ContextVar("ato_trace", default=_NULL)


def current():
    return _current.get()


@contextmanager
def recording(recorder):
    """Record every step run inside this block (on this thread) into recorder."""
    token = _current.set(recorder)
    try:
        yield recorder
    finally:
        _current.reset(token)


def recorder_for(folder: Path, fid: str, round: int = 0):
    """A Recorder, or a NullRecorder if the existing trace cannot be read."""
    try:
        return Recorder(folder, fid, round)
    except Exception:
        _log.warning("trace: recording disabled for %s", fid, exc_info=True)
        return _NULL


# ---- items ---------------------------------------------------------------------------------------

def item(kind: str, obj, *, origin: str, fate: str, rule: str | None = None, fate_rule: str | None = None,
         sources=(), detail: dict | None = None) -> dict:
    """One issue, question, answer or update as it appears in a node."""
    entry = {"type": kind, "origin": origin, "fate": fate, "rule": rule, "fate_rule": fate_rule,
             "from": [s for s in sources if s], "detail": detail or {}}
    if kind == "issue":
        entry.update(key=obj.key, text=obj.description, kind=obj.kind, blocking=obj.blocking, field=obj.field)
    elif kind == "question":
        entry.update(key=obj.key, text=obj.prompt, why=obj.why, issue_keys=list(obj.issue_keys),
                     answer_type=obj.answer_type, example_answer=obj.example_answer)
    return entry


def origin_of(key: str) -> str:
    # Until Issue has an origin field, the code marks its own items with the solver_ prefix.
    return "code" if key.startswith("solver_") else "model"


# ---- pipeline steps (each a no-op outside recording(), and never raising) ------------------------

@guarded
def record_call(call: dict) -> None:
    recorder = current()
    if recorder.enabled:
        recorder.call = call


@guarded
def next_round(round: int) -> None:
    recorder = current()
    if recorder.enabled:
        recorder.round = round


@guarded
def start(label: str) -> None:
    recorder = current()
    if recorder.enabled:
        recorder.node("run.start", "code", "ui.api", extra={"label": label})


@guarded
def commit() -> None:
    recorder = current()
    if recorder.enabled:
        recorder.node("run.commit", "code", "ui.api", parents=[recorder.last.get("present")])


@guarded
def abort(code: str) -> None:
    recorder = current()
    if recorder.enabled:
        recorder.node("run.abort", "code", "ui.api", parents=[recorder.latest], status="failed", failure_code=code)


@guarded
def benchmark(spec, benchmark_id: str) -> None:
    recorder = current()
    if recorder.enabled:
        recorder.spec_node = recorder.node(
            "benchmark", "code", "lbracket.reference_spec", parents=[recorder.last.get("run.start")],
            outputs={"spec": recorder.blob(spec.model_dump(mode="json"))}, extra={"benchmark_id": benchmark_id})


@guarded
def parsed(spec) -> None:
    recorder = current()
    if not recorder.enabled:
        return
    call = recorder.model_record(recorder.take_call("parse"))
    outputs = {**call.pop("outputs", {}), "spec": recorder.blob(spec.model_dump(mode="json"))}
    recorder.spec_node = recorder.node("parse", "model", "ai.parse_problem",
                                       parents=[recorder.last.get("run.start")], outputs=outputs, **call)


@guarded
def critic(review) -> None:
    recorder = current()
    if not recorder.enabled:
        return
    items = [item("issue", i, origin="model", fate="created") for i in review.issues]
    items += [item("question", q, origin="model", fate="created") for q in review.questions]
    recorder.node("critic", "model", "ai.review_problem", parents=[recorder.spec_node], items=items,
                  **recorder.model_record(recorder.take_call("critic")))


@guarded
def resolved(resolution) -> None:
    recorder = current()
    if not recorder.enabled:
        return
    items = [{"type": "update", "origin": "model", "fate": "created", "key": "/".join(map(str, u.path)),
              "text": json.dumps(u.value, default=str), "from": [], "detail": {"path": list(u.path)}}
             for u in resolution.updates]
    recorder.claimed = list(resolution.resolved_issue_keys)
    # A retried resolve reuses the answers recorded by the earlier run.
    answer = recorder.last.get("answer") or recorder.before.get("answer")
    recorder.node("resolve", "model", "ai.resolve_answers", parents=[answer], items=items,
                  extra={"claimed_resolved": recorder.claimed},
                  **recorder.model_record(recorder.take_call("resolve")))


def _at(data, path):
    for part in path:
        try:
            data = data[part]
        except (KeyError, IndexError, TypeError):
            return "<missing>"
    return data


@guarded
def applied(spec, resolution, new_spec, error: Exception | None = None) -> None:
    """Record the edits as applied (new_spec given) or rejected as a whole (new_spec None).

    For a rejection, error may name the offending update (index) and the rule it broke.
    """
    recorder = current()
    if not recorder.enabled:
        return
    before, after = spec.model_dump(mode="json"), new_spec.model_dump(mode="json") if new_spec else None
    resolve = recorder.last.get("resolve")
    culprit, culprit_rule = getattr(error, "index", None), getattr(error, "rule", None)
    items = []
    for n, update in enumerate(resolution.updates):
        if new_spec:
            fate, rule = "applied", None
        elif n == culprit:
            fate, rule = "rejected", culprit_rule
        else:
            fate, rule = "rejected", culprit_rule if culprit is None and culprit_rule else "apply.atomic"
        items.append({"type": "update", "origin": "model", "key": "/".join(map(str, update.path)),
                      "text": json.dumps(update.value, default=str), "fate": fate, "fate_rule": rule,
                      "from": [f"{resolve}#{n}"] if resolve else [],
                      "detail": {"path": list(update.path), "old": recorder.small(_at(before, update.path)),
                                 "new": recorder.small(_at(after, update.path)) if after else None}})
    outputs = ({"spec": recorder.blob(after)} if new_spec
               else {"rejected": recorder.blob(resolution.model_dump(mode="json"))})
    node = recorder.node("apply", "code", "workflow.apply_resolution", parents=[resolve, recorder.spec_node],
                         status="ok" if new_spec else "failed",
                         failure_code=None if new_spec else "invalid_patch", outputs=outputs, items=items)
    if new_spec:
        recorder.spec_node = node


@guarded
def failure(stage: str, failure) -> None:
    recorder = current()
    if not recorder.enabled:
        return
    step = {"review": "critic"}.get(stage, stage)
    recorder.node("failure", "code", "workflow._failed", parents=[recorder.latest], status="failed",
                  failure_code=failure.code,
                  extra={"stage": stage, "request_state": failure.request_state, "retryable": failure.retryable},
                  **recorder.model_record(recorder.take_call(step)))


@guarded
def merged(assessment, review, result, fates, *, interim: bool) -> None:
    """Record the code check and the merge that produced result from assessment and review.

    fates maps ("issue"|"question", index into review) -> (fate, rule) for the model's items.
    """
    recorder = current()
    if not recorder.enabled:
        return
    assess = recorder.node("assess", "code", "lbracket.assess_spec", parents=[recorder.spec_node],
                           interim=interim, items=assessment.trace_items,
                           extra={"config_built": assessment.config is not None})
    critic_node = None if interim else recorder.last.get("critic")
    index = assessment.trace_index

    def code_source(obj):
        return [f"{assess}#{index[id(obj)]}"] if assess and id(obj) in index else []

    def model_source(kind, n):
        offset = 0 if kind == "issue" else len(review.issues)
        return [f"{critic_node}#{offset + n}"] if critic_node else []

    items = [item("issue", i, origin="code", fate="kept", sources=code_source(i)) for i in assessment.issues]
    kept = iter(result.issues[len(assessment.issues):])
    for n, issue in enumerate(review.issues):
        fate, rule = fates[("issue", n)]
        if fate != "dropped":
            items.append(item("issue", next(kept), origin="model", fate=fate, fate_rule=rule,
                              sources=model_source("issue", n)))
    items += [item("question", q, origin="code", fate="kept", sources=code_source(q)) for q in assessment.questions]
    kept = iter(result.questions[len(assessment.questions):])
    for n, question in enumerate(review.questions):
        fate, rule = fates[("question", n)]
        if fate == "kept":
            items.append(item("question", next(kept), origin="model", fate=fate, sources=model_source("question", n)))
    # The kept items come first, in merged order, so later nodes can index into them.
    recorder.origins = [entry["origin"] for entry in items]
    for n, issue in enumerate(review.issues):
        fate, rule = fates[("issue", n)]
        if fate == "dropped":
            items.append(item("issue", issue, origin="model", fate=fate, fate_rule=rule,
                              sources=model_source("issue", n)))
    for n, question in enumerate(review.questions):
        fate, rule = fates[("question", n)]
        if fate != "kept":
            items.append(item("question", question, origin="model", fate=fate, fate_rule=rule,
                              sources=model_source("question", n)))
    extra = {}
    if not interim and recorder.claimed is not None:
        # Display only: which issues the resolver said it resolved are still open after this merge.
        still_open = sorted(set(recorder.claimed) & {i.key for i in result.issues})
        extra["resolved_check"] = {"claimed": recorder.claimed, "still_open": still_open}
        recorder.claimed = None
    recorder.node("merge", "code", "lbracket.combine_review", parents=[assess, critic_node], interim=interim,
                  items=items, extra=extra)
