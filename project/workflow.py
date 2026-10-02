from project.ai import parse_problem, resolve_answers, review_problem
from project.ai_errors import ModelCallError, classify_error
from project.lbracket import assess_spec, combine_review
from project.models import ModelFailure, ProblemSpec, Resolution, Review, Revision, Session


def _index(container, index):
    if type(index) is not int or not isinstance(container, list) or not 0 <= index < len(container):
        raise ValueError("Update contains an invalid list index")
    return container[index]


def apply_resolution(spec: ProblemSpec, resolution: Resolution) -> ProblemSpec:
    """Apply edits atomically and validate the whole typed specification.

    New dictionary keys are allowed only in the two schema-declared maps of
    unit-bearing Values. Unknown model fields and negative indexes are rejected.
    """
    data = spec.model_dump()
    for update in resolution.updates:
        if not update.path:
            raise ValueError("Update path cannot be empty")
        current = data
        for part in update.path[:-1]:
            if type(part) is int:
                current = _index(current, part)
            elif isinstance(part, str) and isinstance(current, dict) and part in current:
                current = current[part]
            else:
                raise ValueError("Update contains an invalid field path")
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
            raise ValueError("Update contains an invalid field path")
    return ProblemSpec.model_validate(data)


def parse_succeeded(session: Session) -> bool:
    """Recognize old saved sessions without inventing a successful parse."""
    if session.parse_completed is not None:
        return session.parse_completed
    if any(item.step == "parse" for item in session.usage):
        return True
    return session.spec != ProblemSpec(name="Unparsed request")


def deterministic_blockers(session: Session) -> list[str]:
    # An empty placeholder after an API failure is not evidence that the
    # engineer omitted geometry, loads, or material from their request.
    if not parse_succeeded(session):
        return []
    return [issue.description for issue in assess_spec(session.spec).issues if issue.blocking]


def ready_to_try(session: Session) -> bool:
    if session.failure is not None or not parse_succeeded(session):
        return False
    return assess_spec(session.spec).ready and not any(issue.blocking for issue in session.review.issues)


def _failed(session: Session, step: str, error: Exception) -> Session:
    # SDK exceptions can contain headers or response fragments. Persist only
    # controlled wording, never exception text or provider payloads.
    error = classify_error(error)
    failure = ModelFailure(
        stage=step, code=error.code, message=error.message,
        next_step=error.next_step, request_state=error.request_state,
        retryable=error.retryable,
    )
    message = f"{step}: {failure.message} {failure.next_step} No solver was launched."
    return session.model_copy(update={
        "errors": [*session.errors, message],
        "failure": failure,
    })


def start_session(request: str, context: str | None = None) -> Session:
    fallback = Session(original_request=request, context=context,
                       spec=ProblemSpec(name="Unparsed request"), review=Review(),
                       parse_completed=False)
    try:
        spec, parse_usage = parse_problem(request, context)
    except Exception as error:
        return _failed(fallback, "parse", error)
    session = Session(original_request=request, context=context, spec=spec,
                      review=combine_review(spec, Review()), usage=[parse_usage],
                      parse_completed=True)
    try:
        review, review_usage = review_problem(spec, original_request=request, context=context, revisions=[])
    except Exception as error:
        return _failed(session, "review", error)
    return session.model_copy(update={"review": combine_review(spec, review),
                                      "usage": [parse_usage, review_usage]})


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
            # A schema-valid response can still contain an invalid field path.
            # Retrying must request fresh changes, even if a subsequent network
            # error temporarily replaces the original invalid-patch failure.
            bypass_cache=bool(session.failure and session.failure.stage == "resolve"),
        )
    except Exception as error:
        return _failed(session, "resolve", error)
    try:
        new_spec = apply_resolution(session.spec, resolution)
    except Exception:
        return _failed(session, "resolve", ModelCallError(
            code="invalid_schema",
            message=("The saved Claude response contains changes that could not be applied to the specification."
                     if resolve_usage.cache_hit else
                     "Claude returned changes that could not be applied to the specification."),
            next_step="Retry to request fresh changes from Claude. Your answers and last valid specification have been kept.",
            request_state="not_sent" if resolve_usage.cache_hit else "response_received", retryable=True,
        ))
    revision = Revision(questions=session.review.questions, answers=answers, resolution=resolution)
    revisions = [*session.revisions, revision]
    candidate = Session(original_request=session.original_request, context=session.context,
                        spec=new_spec, review=combine_review(new_spec, Review()),
                        revisions=revisions, usage=[*session.usage, resolve_usage], errors=session.errors,
                        parse_completed=True)
    try:
        review, review_usage = review_problem(new_spec, original_request=session.original_request,
                                             context=session.context, revisions=revisions)
    except Exception as error:
        return _failed(candidate, "review", error)
    return candidate.model_copy(update={"review": combine_review(new_spec, review),
                                        "usage": [*candidate.usage, review_usage]})


def _retry_review(session: Session) -> Session:
    candidate = Session.model_validate(session.model_dump())
    try:
        review, review_usage = review_problem(candidate.spec,
                                             original_request=candidate.original_request,
                                             context=candidate.context, revisions=candidate.revisions)
    except Exception as error:
        return _failed(candidate, "review", error)
    return candidate.model_copy(update={"review": combine_review(candidate.spec, review),
                                        "usage": [*candidate.usage, review_usage],
                                        "failure": None, "parse_completed": True})


def retry_failure(session: Session) -> Session:
    """Resume the failed stage while preserving successful work and audit history."""
    if not parse_succeeded(session) or (session.failure and session.failure.stage == "parse"):
        recovered = start_session(session.original_request, session.context)
        return recovered.model_copy(update={
            "errors": [*session.errors, *recovered.errors],
            "usage": [*session.usage, *recovered.usage],
            "revisions": session.revisions,
        })
    if session.failure and session.failure.stage == "resolve":
        return answer_questions(session, session.pending_answers)
    return _retry_review(session)


def retry_review(session: Session) -> Session:
    """Compatibility entry point; retry the stage that actually failed."""
    return retry_failure(session)
