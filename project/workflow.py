from project.ai import parse_problem, resolve_answers, review_problem
from project.lbracket import assess_spec, combine_review
from project.models import Issue, ProblemSpec, Resolution, Review, Revision, Session


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


def deterministic_blockers(session: Session) -> list[str]:
    return [issue.description for issue in assess_spec(session.spec).issues if issue.blocking]


def ready_to_try(session: Session) -> bool:
    return assess_spec(session.spec).ready and not any(issue.blocking for issue in session.review.issues)


def _failed(session: Session, step: str) -> Session:
    # SDK exceptions can contain headers or response fragments. Persist only
    # controlled wording, never exception text or provider payloads.
    message = f"The {step} step failed. The last valid specification was kept. Check the API setup or retry; no solver was launched."
    return session.model_copy(update={
        "errors": [*session.errors, message],
        "review": Review(issues=[*session.review.issues,
                                Issue(key="formulation_error", description=message, kind="data")],
                         questions=session.review.questions),
    })


def start_session(request: str, context: str | None = None) -> Session:
    fallback = Session(original_request=request, context=context,
                       spec=ProblemSpec(name="Unparsed request"), review=Review())
    try:
        spec, parse_usage = parse_problem(request, context)
    except Exception:
        return _failed(fallback, "parse")
    session = Session(original_request=request, context=context, spec=spec,
                      review=combine_review(spec, Review()), usage=[parse_usage])
    try:
        review, review_usage = review_problem(spec, original_request=request, context=context, revisions=[])
    except Exception:
        return _failed(session, "review")
    return session.model_copy(update={"review": combine_review(spec, review),
                                      "usage": [parse_usage, review_usage]})


def answer_questions(session: Session, answers: dict[str, str]) -> Session:
    session = Session.model_validate(session.model_dump())
    try:
        resolution, resolve_usage = resolve_answers(
            session.spec, session.review, answers,
            original_request=session.original_request, context=session.context,
            revisions=session.revisions,
        )
        new_spec = apply_resolution(session.spec, resolution)
    except Exception:
        return _failed(session, "answer resolution")
    revision = Revision(questions=session.review.questions, answers=answers, resolution=resolution)
    revisions = [*session.revisions, revision]
    candidate = Session(original_request=session.original_request, context=session.context,
                        spec=new_spec, review=combine_review(new_spec, Review()),
                        revisions=revisions, usage=[*session.usage, resolve_usage], errors=session.errors)
    try:
        review, review_usage = review_problem(new_spec, original_request=session.original_request,
                                             context=session.context, revisions=revisions)
    except Exception:
        return _failed(candidate, "review")
    return candidate.model_copy(update={"review": combine_review(new_spec, review),
                                        "usage": [*candidate.usage, review_usage]})


def retry_review(session: Session) -> Session:
    """Retry a transient review error without reparsing or losing prior answers."""
    candidate = Session.model_validate(session.model_dump())
    try:
        review, review_usage = review_problem(candidate.spec,
                                             original_request=candidate.original_request,
                                             context=candidate.context, revisions=candidate.revisions)
    except Exception:
        return _failed(candidate, "review")
    return candidate.model_copy(update={"review": combine_review(candidate.spec, review),
                                        "usage": [*candidate.usage, review_usage]})
