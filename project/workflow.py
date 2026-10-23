from project import trace
from project.ai import parse_problem, resolve_answers, review_problem
from project.ai_errors import ModelCallError, classify_error
from project.lbracket import ASSUMPTION_OPTIONS, MAX_QUESTIONS, assess_spec, combine_review, start_issues
from project.models import CODE_KEY_PREFIX, ModelFailure, ProblemSpec, Resolution, Review, Revision, Session


UNPARSED_NAME = "Unparsed request"   # name of the placeholder spec before a successful parse
ENGINEER_EDIT_KEY = "engineer_edit"   # answer key of a round in which the engineer set values directly


class PatchRejected(ValueError):
    """An edit set that cannot be applied; index and rule name the first update at fault (for the trace)."""

    def __init__(self, message: str, rule: str, index: int | None = None):
        super().__init__(message)
        self.rule, self.index = rule, index


def _index(container, index):
    if type(index) is not int or not isinstance(container, list) or not 0 <= index < len(container):
        raise PatchRejected("Update contains an invalid list index", "apply.index_bounds")
    return container[index]


def apply_resolution(spec: ProblemSpec, resolution: Resolution) -> ProblemSpec:
    """Apply edits atomically and validate the whole typed specification.

    New dictionary keys are allowed only in the two schema-declared maps of
    unit-bearing Values. Unknown model fields and negative indexes are rejected.
    """
    data = spec.model_dump()
    for n, update in enumerate(resolution.updates):
        try:
            _apply(data, update)
        except PatchRejected as error:
            error.index = n
            raise
    try:
        return ProblemSpec.model_validate(data)
    except ValueError as error:
        raise PatchRejected("The edited specification is not valid", "apply.revalidate") from error


def _apply(data: dict, update) -> None:
    if not update.path:
        raise PatchRejected("Update path cannot be empty", "apply.empty_path")
    current = data
    for part in update.path[:-1]:
        if type(part) is int:
            current = _index(current, part)
        elif isinstance(part, str) and isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise PatchRejected("Update contains an invalid field path", "apply.path_exists")
    last = update.path[-1]
    parent = update.path[:-1]
    value_map = parent == ["geometry", "parameters"] or (
        len(parent) == 3 and parent[0] == "materials" and type(parent[1]) is int and parent[2] == "properties"
    )
    if type(last) is int:
        _index(current, last)
        current[last] = update.value
    elif isinstance(last, str) and isinstance(current, dict) and (last in current or value_map):
        current[last] = update.value
    else:
        raise PatchRejected("Update contains an invalid field path", "apply.new_key_value_maps_only"
                            if isinstance(current, dict) else "apply.path_exists")


def _code_check(spec: ProblemSpec, kept: frozenset[str] = frozenset()) -> dict:
    """What the reviewer is told the code check already found, and how many questions it may add."""
    assessment = assess_spec(spec, kept)
    return {"code_issues": [issue.model_dump() for issue in assessment.issues],
            "free_question_slots": max(0, MAX_QUESTIONS - len(assessment.questions))}


def kept_assumptions(session: Session) -> frozenset[str]:
    """Assumption questions the engineer kept: answered with "keep" (lbracket.ASSUMPTION_OPTIONS[1]), or with
    their own words after which the resolver left `assumptions` as it was. Asked once, then reported unsupported
    (an assumption the resolver removed is no longer in the spec, so it is not asked either way)."""
    kept = set()
    for revision in session.revisions:
        touched = any(update.path and update.path[0] == "assumptions" for update in revision.resolution.updates)
        for key, answer in revision.answers.items():
            if key.startswith(CODE_KEY_PREFIX + "assumption_") and (
                    answer == ASSUMPTION_OPTIONS[1] or (answer not in ASSUMPTION_OPTIONS and not touched)):
                kept.add(key)
    return frozenset(kept)


def parse_succeeded(session: Session) -> bool:
    return session.parse_completed


def ready_to_try(session: Session) -> bool:
    if session.failure is not None or not parse_succeeded(session):
        return False
    assessment = assess_spec(session.spec)
    # The dry run (ParaLeSTO in a child process) goes last: only a spec nothing else blocks reaches it.
    return (assessment.ready and not any(issue.blocking for issue in session.review.issues)
            and not start_issues(assessment.config))


def _failed(session: Session, step: str, error: Exception) -> Session:
    # SDK exceptions can contain headers or response fragments. Persist only
    # controlled wording, never exception text or provider payloads.
    error = classify_error(error)
    failure = ModelFailure(
        stage=step, code=error.code, message=error.message,
        next_step=error.next_step, request_state=error.request_state,
        retryable=error.retryable,
    )
    trace.failure(step, failure)
    return session.model_copy(update={"failure": failure})


def start_session(request: str, context: str | None = None) -> Session:
    fallback = Session(original_request=request, context=context,
                       spec=ProblemSpec(name=UNPARSED_NAME), review=Review(),
                       parse_completed=False)
    try:
        spec, parse_usage = parse_problem(request, context)
    except Exception as error:
        return _failed(fallback, "parse", error)
    trace.parsed(spec)
    session = Session(original_request=request, context=context, spec=spec,
                      review=combine_review(spec, Review(), interim=True), usage=[parse_usage],
                      parse_completed=True)
    return _review(session)


def copy_session(session: Session) -> Session:
    """The session of a copy of an approved formulation: its spec, request and rounds, with the reviewer's items
    merged again with a fresh code check. No model call; model usage and clarifications stay with the original."""
    session = Session.model_validate(session.model_dump())
    copy = Session(original_request=session.original_request, context=session.context, spec=session.spec,
                   revisions=session.revisions, parse_completed=True, review=Review())
    return copy.model_copy(update={"review": combine_review(copy.spec, session.review, kept=kept_assumptions(copy))})


def answer_questions(session: Session, answers: dict[str, str]) -> Session:
    session = Session.model_validate(session.model_dump())
    if not parse_succeeded(session):
        # The request must first be parsed before answers can update its spec.
        return session
    session = session.model_copy(update={"pending_answers": dict(answers)})
    try:
        resolution, resolve_usage = resolve_answers(
            session.spec, session.review, answers,
            original_request=session.original_request, context=session.context,
            revisions=session.revisions,
            clarifications=[c.model_dump(exclude={"at", "failure"}) for c in session.clarifications
                            if c.round == len(session.revisions) and c.failure is None],
            # A schema-valid response can still contain an invalid field path.
            # Retrying must request fresh changes, even if a subsequent network
            # error temporarily replaces the original invalid-patch failure.
            bypass_cache=bool(session.failure and session.failure.stage == "resolve"),
        )
    except Exception as error:
        return _failed(session, "resolve", error)
    trace.resolved(resolution)
    try:
        new_spec = apply_resolution(session.spec, resolution)
    except Exception as error:
        trace.applied(session.spec, resolution, None, error)
        return _failed(session, "resolve", ModelCallError(
            code="invalid_schema",
            message=("The saved model response contains changes that could not be applied to the specification."
                     if resolve_usage.cache_hit else
                     "The model returned changes that could not be applied to the specification."),
            next_step="Retry to request fresh changes from the model. Your answers and last valid specification have been kept.",
            request_state="not_sent" if resolve_usage.cache_hit else "response_received", retryable=True,
        ))
    trace.applied(session.spec, resolution, new_spec)
    return _next_round(session, new_spec, Revision(questions=session.review.questions, answers=answers,
                                                   resolution=resolution), [*session.usage, resolve_usage])


def apply_edits(session: Session, resolution: Resolution, note: str) -> Session:
    """Values the engineer set directly: applied by code with no model call, recorded as a round whose
    answer (ENGINEER_EDIT_KEY) lists them, then checked and reviewed like answers are.

    ui.api.edit_values checks the edits apply before starting this step.
    """
    session = Session.model_validate(session.model_dump())
    trace.edited(session.spec, resolution)
    try:
        new_spec = apply_resolution(session.spec, resolution)
    except PatchRejected as error:
        trace.applied(session.spec, resolution, None, error, by="edit")
        raise
    trace.applied(session.spec, resolution, new_spec, by="edit")
    return _next_round(session, new_spec, Revision(questions=session.review.questions,
                                                   answers={ENGINEER_EDIT_KEY: note}, resolution=resolution),
                       session.usage)


def _next_round(session: Session, new_spec: ProblemSpec, revision: Revision, usage: list) -> Session:
    """The session after a round's edits were applied: the new spec, the code check, then the reviewer."""
    revisions = [*session.revisions, revision]
    trace.next_round(len(revisions))
    candidate = Session(original_request=session.original_request, context=session.context,
                        spec=new_spec, revisions=revisions, usage=usage,
                        clarifications=session.clarifications, parse_completed=True, review=Review())
    candidate = candidate.model_copy(update={
        "review": combine_review(new_spec, Review(), interim=True, kept=kept_assumptions(candidate))})
    return _review(candidate)


def _review(session: Session) -> Session:
    """Run the reviewer on the session's spec and merge its review with the code check."""
    candidate = Session.model_validate(session.model_dump())
    kept = kept_assumptions(candidate)
    try:
        review, review_usage = review_problem(candidate.spec,
                                             original_request=candidate.original_request,
                                             context=candidate.context, revisions=candidate.revisions,
                                             **_code_check(candidate.spec, kept))
    except Exception as error:
        return _failed(candidate, "review", error)
    trace.critic(review)
    return candidate.model_copy(update={"review": combine_review(candidate.spec, review, kept=kept),
                                        "usage": [*candidate.usage, review_usage],
                                        "failure": None, "parse_completed": True})


def retry_failure(session: Session) -> Session:
    """Resume the failed stage while preserving successful work and audit history."""
    if not parse_succeeded(session) or (session.failure and session.failure.stage == "parse"):
        recovered = start_session(session.original_request, session.context)
        return recovered.model_copy(update={
            "usage": [*session.usage, *recovered.usage],
            "revisions": session.revisions,
        })
    if session.failure and session.failure.stage == "resolve":
        return answer_questions(session, session.pending_answers)
    return _review(session)
