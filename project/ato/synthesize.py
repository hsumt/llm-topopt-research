"""Mid-run tool synthesis: write the edit operation the hypothesis needs.

When ``capability.py`` reports a gap, this stage asks a model to author a new
``EditOperation`` -- source code -- which is then admitted to the run only if it
passes a static gate and a smoke test. This is what stops the hypothesis space
from being capped by whatever the toolbox happened to contain on the day.

What a synthesized operation is allowed to be
---------------------------------------------
**A pure mutation of the specification object. Nothing else.** It may assign to
fields of ``spec``, construct a ``LoadCase``, and append to ``spec.load_cases``. It
may not import anything outside the specification module, call anything that
touches the filesystem, network or interpreter, compute a physical quantity, or
return a value.

That restriction is the safety property of the whole design, and it is worth
stating why rather than only enforcing it. The verdict on a hypothesis comes from
re-solving the modified problem with the same verified solver. If a written
operation could compute, fetch or assert anything, an agent would be able to
influence its own verdict -- writing the exam and marking it. Confining it to
"change the stated problem, then stand back" keeps generation and adjudication
apart no matter what the model writes.

Two further properties are not negotiable and are not delegated:

* **Tier** is assigned by ``grammar.tier_for_slot`` from the slot and the declared
  intent. The writer's suggestion is recorded and overridden. A stage that could
  set its own admissibility tier could route an intent-altering edit past the
  authority gate.
* **Persistence.** A synthesized operation lives for the duration of the run, in
  ``grammar.SYNTHESIZED``. Its source is recorded in the run record for review; it
  is never written back into the committed grammar by the system itself.
"""
from __future__ import annotations

import ast
import copy
import json
from dataclasses import fields as dataclass_fields

from project.ato.grammar import (EditOperation, TIER_ORDER, register_synthesized,
                                 tier_for_slot)
from project.topopt.stress import problem as P

# --- static gate -------------------------------------------------------------

_ALLOWED_NODES = {
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Assign, ast.AugAssign,
    ast.Expr, ast.Call, ast.Attribute, ast.Name, ast.Load, ast.Store, ast.Constant,
    ast.Subscript, ast.List, ast.Tuple, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
    ast.If, ast.IfExp, ast.For, ast.Return, ast.ImportFrom, ast.alias, ast.keyword,
    ast.UnaryOp, ast.USub, ast.UAdd, ast.Not, ast.BoolOp, ast.And, ast.Or, ast.Pass,
}
_ALLOWED_CALLS = {"float", "int", "bool", "str", "list", "tuple", "len", "min", "max",
                  "abs", "round", "range", "enumerate", "LoadCase"}
_ALLOWED_METHODS = {"append", "extend", "copy"}
_ALLOWED_IMPORT_MODULE = "project.topopt.stress.problem"
_ALLOWED_IMPORT_NAMES = {"LoadCase", "MODELS", "Geometry", "Material", "Constraints",
                         "Optimizer", "ModelCannotRepresent"}
_FORBIDDEN = {
    "__import__", "eval", "exec", "compile", "open", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "hasattr", "input", "breakpoint", "exit", "quit",
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib", "builtins",
    "__builtins__", "requests", "urllib", "pickle", "marshal", "ctypes",
}


class Params(dict):
    """Parameter mapping that supports both ``prm["x"]`` and ``prm.x``.

    Written operations reach for attribute access on the parameter mapping about as
    often as subscripting, and a plain dict turns that into an ``AttributeError``
    that reads like a code defect rather than a style mismatch. Measured: two of
    three synthesis attempts in one run were rejected for exactly this. Accepting
    both is free -- reading an attribute off a mapping grants no capability the
    static gate was withholding.
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(
                f"no parameter named {key!r}; declared parameters are {sorted(self)}"
            ) from exc


class SynthesisRejected(RuntimeError):
    """The written operation failed the gate or the smoke test."""


def _root_name(node) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def gate_source(source: str) -> ast.Module:
    """Static admissibility check. Raises ``SynthesisRejected`` with every reason."""
    errs = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SynthesisRejected(f"source does not parse: {exc}") from exc

    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    non_func = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.ImportFrom))]
    if len(funcs) != 1 or funcs[0].name != "apply":
        errs.append("source must define exactly one function, named 'apply'")
    if non_func:
        errs.append("only a single 'apply' function and imports from the specification "
                    "module are allowed at module level")
    if funcs:
        args = [a.arg for a in funcs[0].args.args]
        if args != ["spec", "prm"]:
            errs.append(f"apply must take exactly (spec, prm); got {args}")

    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            errs.append(f"disallowed syntax {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN:
            errs.append(f"forbidden name {node.id!r}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            errs.append(f"dunder attribute access {node.attr!r}")
        if isinstance(node, ast.ImportFrom):
            if node.module != _ALLOWED_IMPORT_MODULE:
                errs.append(f"import from {node.module!r} is not allowed; only "
                            f"{_ALLOWED_IMPORT_MODULE}")
            for a in node.names:
                if a.name not in _ALLOWED_IMPORT_NAMES:
                    errs.append(f"import of {a.name!r} is not allowed")
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                if f.id not in _ALLOWED_CALLS:
                    errs.append(f"call to {f.id!r} is not allowed")
            elif isinstance(f, ast.Attribute):
                if f.attr not in _ALLOWED_METHODS:
                    errs.append(f"method call .{f.attr}() is not allowed")
                elif _root_name(f) != "spec":
                    errs.append(f"method .{f.attr}() must be called on spec, not "
                                f"{_root_name(f)!r}")
            else:
                errs.append("only direct function or spec-method calls are allowed")
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, (ast.Attribute, ast.Subscript)):
                    if _root_name(t) != "spec":
                        errs.append(f"assignment must target spec.*, not "
                                    f"{_root_name(t)!r}.*")
                elif not isinstance(t, ast.Name):
                    errs.append("unsupported assignment target")
    if errs:
        raise SynthesisRejected("; ".join(dict.fromkeys(errs)))
    return tree


def compile_apply(source: str):
    """Gate, then compile in a namespace with no builtins beyond the allowed calls.

    The whitelisted ``from project.topopt.stress.problem import ...`` statements are
    **stripped from the tree** rather than executed, and the names they would have
    bound are pre-injected instead. That way the execution namespace never receives
    ``__import__`` at all: import machinery is the single most useful primitive for
    escaping a restricted namespace, and there is no reason to hand it over when the
    only legitimate imports are known in advance.
    """
    tree = gate_source(source)
    tree.body = [n for n in tree.body if not isinstance(n, ast.ImportFrom)]
    ast.fix_missing_locations(tree)
    bi = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    safe_builtins = {k: bi[k] for k in
                     ("float", "int", "bool", "str", "list", "tuple", "len",
                      "min", "max", "abs", "round", "range", "enumerate")}
    ns = {"__builtins__": safe_builtins}
    for name in _ALLOWED_IMPORT_NAMES:
        if hasattr(P, name):
            ns[name] = getattr(P, name)
    exec(compile(tree, "<synthesized>", "exec"), ns)   # gated above
    fn = ns.get("apply")
    if not callable(fn):
        raise SynthesisRejected("no callable 'apply' produced")

    def apply_with_params(spec, prm):
        return fn(spec, prm if isinstance(prm, Params) else Params(prm or {}))

    apply_with_params.source = source
    return apply_with_params


# --- schema handed to the writer ---------------------------------------------

def spec_schema() -> dict:
    """Field paths of the specification, generated from the dataclasses.

    Generated rather than written by hand so the writer can never be told about a
    field that does not exist, or miss one that does.
    """
    def block(cls):
        return {f.name: type(f.default).__name__ if f.default is not None else "value"
                for f in dataclass_fields(cls)}
    return {
        "spec.formulation": "one of P1, P2, P3",
        "spec.geometry.*": block(P.Geometry),
        "spec.geometry.model": f"one of {list(P.MODELS)}",
        "spec.material.*": block(P.Material),
        "spec.constraints.*": block(P.Constraints),
        "spec.optimizer.*": block(P.Optimizer),
        "spec.load_cases": ("list of LoadCase(magnitude: float N, distribute_nodes: int, "
                            "direction: 'x'|'y'|'z', name: str); 'z' is out of plane and is "
                            "only representable when spec.geometry.model == 'extruded_3d'"),
    }


WRITER_SYSTEM_PROMPT = """You write a new edit operation for a specification-adequacy \
diagnosis system, in Python, to realise a hypothesis the existing toolbox cannot express.

You are writing a PURE MUTATION OF THE SPECIFICATION OBJECT. That is the whole contract:

    def apply(spec, prm):
        # assign to fields of spec; append to spec.load_cases; nothing else
        ...

HARD LIMITS -- source violating any of these is rejected outright by a static checker:
  * exactly one module-level function, named apply, taking exactly (spec, prm)
  * assignments may only target spec.<field> paths (local variables are fine)
  * the only permitted method calls are .append/.extend/.copy on spec.*
  * the only permitted plain calls are float, int, bool, str, list, tuple, len, min, max, \
abs, round, range, enumerate, LoadCase
  * the only permitted import is `from project.topopt.stress.problem import LoadCase` \
(and other names from that module)
  * no filesystem, network, subprocess, eval/exec, getattr/setattr, dunder access
  * do not compute a physical quantity, do not return anything

`prm` is the parameter mapping; both prm["name"] and prm.name work. Use only parameters you
declared. Access fields as spec.geometry.thickness (attributes), never spec["geometry"].

WHY the limits are what they are: whether the hypothesis is correct is decided by re-solving \
the modified problem with a verified solver. If your operation could compute or assert \
anything, the system would be marking its own exam. Change the stated problem and stand back.

You must also declare the slot the edit belongs to, its parameters with numeric bounds, and \
whether the specification determines those parameters (BOUND) or not (UNBOUND). Propose an \
admissibility tier if you wish; it is recorded and then reassigned from the slot and the \
declared intent, so do not rely on it.

If realising the hypothesis needs TWO independent changes that an engineer would sign off on \
separately, write the operation for ONE of them and say so in `note`. The subset layer will \
combine it with others; a single operation bundling several unrelated changes cannot be \
attributed if it works."""

WRITER_TOOL = {
    "name": "write_operation",
    "description": "Author a new specification edit operation as Python source.",
    "input_schema": {
        "type": "object",
        "properties": {
            "operation_name": {"type": "string",
                               "description": "snake_case, verb-first, e.g. 'add_lateral_load_case'"},
            "slot": {"type": "string",
                     "description": "problem-tuple slot, e.g. 'load cases', "
                                    "'discretisation (analysis model)', 'design domain (geometry)'"},
            "description": {"type": "string"},
            "apply_source": {"type": "string",
                             "description": "Python source defining exactly def apply(spec, prm)"},
            "parameters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "kind": {"type": "string", "enum": ["int", "float"]},
                        "low": {"type": "number"},
                        "high": {"type": "number"},
                        "probe_value": {"type": "number",
                                        "description": "a defensible value to test the operation with"},
                        "description": {"type": "string"},
                    },
                    "required": ["name", "kind", "low", "high", "probe_value"],
                },
            },
            "boundedness": {"type": "string", "enum": ["BOUND", "UNBOUND"]},
            "parameter_source": {"type": "string"},
            "proposed_tier": {"type": "string", "enum": ["T1", "T2", "T3"]},
            "loosens_a_stated_requirement": {
                "type": "boolean",
                "description": "true if the edit makes a stated limit easier to satisfy"},
            "note": {"type": "string"},
        },
        "required": ["operation_name", "slot", "description", "apply_source",
                     "parameters", "boundedness", "parameter_source"],
    },
}


#: Specification field path -> problem-tuple slot. Used to work out what a written
#: operation ACTUALLY touches, rather than trusting the slot it declares.
FIELD_SLOTS = (
    ("geometry.model", "discretisation (analysis model)"),
    ("geometry.n_layers", "discretisation (analysis model)"),
    ("geometry.n_cells_per_side", "discretisation"),
    ("geometry.exclude", "non-design region"),
    ("geometry.thickness", "design domain (geometry)"),
    ("geometry.corner_fillet_radius", "design domain (geometry)"),
    ("geometry.arm_fraction", "design domain (geometry)"),
    ("geometry.L", "design domain (geometry)"),
    ("load_cases", "load cases"),
    ("constraints.stress_limit", "inequality constraints"),
    ("constraints.mass_fraction_limit", "inequality constraints"),
    ("constraints.n_clusters", "regularisation"),
    ("constraints.p_norm", "regularisation"),
    ("constraints.clustering", "regularisation"),
    ("constraints.recluster_every", "regularisation"),
    ("optimizer.", "regularisation"),
    ("formulation", "objective"),
    ("material.", "inequality constraints"),
)


def slot_of_field(path: str) -> str:
    for prefix, slot in FIELD_SLOTS:
        if path.startswith(prefix):
            return slot
    return "unclassified"


def changed_fields(before: dict, after: dict, prefix: str = "") -> list:
    """Dotted paths of every field the operation altered."""
    out = []
    keys = set(before) | set(after)
    for k in sorted(keys):
        b, a = before.get(k), after.get(k)
        path = f"{prefix}{k}"
        if isinstance(b, dict) and isinstance(a, dict):
            out.extend(changed_fields(b, a, prefix=path + "."))
        elif b != a:
            out.append(path)
    return out


def _validator(param_specs):
    def validate(d):
        errs = []
        for ps in param_specs:
            v = d.get(ps["name"])
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                errs.append(f"{ps['name']} must be a number")
                continue
            if ps["kind"] == "int" and float(v) != int(v):
                errs.append(f"{ps['name']} must be an integer")
            if not (ps["low"] <= float(v) <= ps["high"]):
                errs.append(f"{ps['name']}={v} outside [{ps['low']}, {ps['high']}]")
        return errs
    return validate


def smoke_test(fn, base_spec, probe: dict) -> dict:
    """Apply the operation to a coarsened copy and confirm it changes the problem.

    ``ModelCannotRepresent`` is a legitimate outcome, not a failure: it means the
    operation asks the analysis model for something it cannot answer, which is
    exactly the case where the operation only has meaning combined with an enabling
    edit. The subset layer handles that; the smoke test only has to establish that
    the operation is well formed and actually alters the stated problem.
    """
    from project.topopt.stress.solver import Evaluator

    spec = base_spec.copy()
    spec.geometry.n_cells_per_side = 20
    spec.optimizer.max_iter = 1
    before = json.dumps(spec.to_dict(), sort_keys=True, default=str)
    try:
        fn(spec, probe)
    except Exception as exc:
        # The static gate establishes that the source is *shaped* correctly; it
        # cannot establish that the field paths exist or that the types line up.
        # A written operation that raises when actually applied is a rejection, not
        # a crash: synthesis is expected to fail sometimes and the run must carry on
        # with whatever else it has.
        raise SynthesisRejected(
            f"apply() raised when applied to the specification: "
            f"{type(exc).__name__}: {exc}") from exc
    after = json.dumps(spec.to_dict(), sort_keys=True, default=str)
    if before == after:
        raise SynthesisRejected("operation left the specification unchanged")
    paths = changed_fields(json.loads(before), json.loads(after))
    slots = sorted({slot_of_field(p) for p in paths})
    try:
        ev = Evaluator(spec)
        ev.evaluate(ev.initial_design())
        return {"solvable": True, "changed_fields": paths, "slots_touched": slots,
                "note": "modified problem evaluated once successfully"}
    except P.ModelCannotRepresent as exc:
        return {"solvable": False, "not_representable": True,
                "changed_fields": paths, "slots_touched": slots, "note": str(exc)}
    except Exception as exc:
        raise SynthesisRejected(
            f"modified problem could not be evaluated: {type(exc).__name__}: {exc}") from exc


def synthesize(backend, gap: dict, base_spec, intent=None) -> dict:
    """Write, gate, smoke-test and register one operation. Returns a record."""
    h = gap["hypothesis"]
    user = (
        (intent.summary_for_prompt() + "\n\n" if intent is not None else "")
        + "HYPOTHESIS that no existing operation can express:\n"
        + json.dumps(h, indent=2, default=str)
        + f"\n\nMISSING CAPABILITY, as identified by the matcher:\n{gap.get('missing_capability')}\n"
        + f"\nMatcher note: {gap.get('note', '')}\n"
        + "\nSPECIFICATION SCHEMA -- these are the only fields that exist:\n"
        + json.dumps(spec_schema(), indent=2, default=str)
        + "\n\nCURRENT STATED PROBLEM:\n"
        + json.dumps(base_spec.to_dict(), indent=2, default=str)
    )
    raw = backend(WRITER_SYSTEM_PROMPT, user, WRITER_TOOL)
    record = {"hypothesis_id": h.get("id"), "requested_capability": gap.get("missing_capability"),
              "written": raw, "model": getattr(backend, "model", "unknown")}
    try:
        fn = compile_apply(raw["apply_source"])
        param_specs = list(raw.get("parameters") or [])
        # A parameterless operation is legitimate: switching the analysis model, for
        # instance, has nothing to tune. Rejecting it would block exactly the
        # enabling edits this stage exists to write.
        probe = {ps["name"]: (int(ps["probe_value"]) if ps["kind"] == "int"
                              else float(ps["probe_value"])) for ps in param_specs}
        record["probe"] = probe
        record["smoke"] = smoke_test(fn, base_spec, probe)

        # Tier from what the code TOUCHES, not from the slot it declares. A writer
        # that bundles a model switch into a load-case operation -- observed -- would
        # otherwise carry the declared slot's tier while altering something else. Taking
        # the most restrictive tier over every touched slot closes that: an operation
        # cannot lower its own authority requirement by mislabelling itself.
        touched = list(record["smoke"].get("slots_touched") or [raw["slot"]])
        graded = [tier_for_slot(s, intent=intent,
                                loosens=raw.get("loosens_a_stated_requirement"))
                  for s in touched] or [tier_for_slot(raw["slot"], intent=intent)]
        tier, tier_reason = max(graded, key=lambda t: TIER_ORDER.get(t[0], 9))
        declared_tier, _ = tier_for_slot(raw["slot"], intent=intent,
                                         loosens=raw.get("loosens_a_stated_requirement"))
        if len(touched) > 1:
            record["bundled_slots"] = touched
            tier_reason += (f"; the operation touches {len(touched)} slots {touched} and is "
                            f"graded on the most restrictive, not on its declared "
                            f"{raw['slot']!r} ({declared_tier})")
        op = EditOperation(
            name=raw["operation_name"], slot=raw["slot"],
            description=raw["description"], tier=tier,
            boundedness=raw.get("boundedness", "UNBOUND"),
            parameter_source=raw.get("parameter_source", "unstated"),
            parameters={ps["name"]: (f"{ps['kind']}, {ps['low']}..{ps['high']}, "
                                     f"{ps.get('description', '')}") for ps in param_specs},
            apply=fn, validate=_validator(param_specs),
        )
        # Flag a written operation that duplicates something already committed. The
        # matcher called a corner-fillet edit a gap in one run although
        # declare_corner_fillet exists, so duplicates are real and worth recording --
        # they are evidence about matcher quality, not a reason to reject the tool.
        from project.ato.grammar import OPERATIONS as _COMMITTED
        dup = [n for n, o in _COMMITTED.items()
               if o.slot == op.slot and set(o.parameters) & set(op.parameters)]
        if dup:
            record["possible_duplicate_of"] = dup
        register_synthesized(op)
        record.update({"admitted": True, "operation": op.name, "tier": tier,
                       "slots_touched": touched, "declared_slot": raw["slot"],
                       "tier_reason": tier_reason,
                       "proposed_tier": raw.get("proposed_tier"),
                       "tier_agreed_with_writer": raw.get("proposed_tier") == tier,
                       "candidate": {"id": f"{h.get('id')}S", "operation": op.name,
                                     "parameters": probe,
                                     "targets_anomaly": h.get("targets_anomaly", []),
                                     "rationale": h.get("rationale", ""),
                                     "origin": "synthesized", "hypothesis": h}})
    except SynthesisRejected as exc:
        record.update({"admitted": False, "rejected_because": str(exc)})
    except Exception as exc:   # defence in depth: never let synthesis kill a run
        record.update({"admitted": False,
                       "rejected_because": f"unexpected {type(exc).__name__}: {exc}"})
    return record
