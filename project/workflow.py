from project.ai import (
    parse_problem,
    resolve_answers,
    review_problem,
)
from project.models import (
    ProblemSpec,
    Resolution,
    Revision,
    Session,
)


def apply_resolution(
    spec: ProblemSpec,
    resolution: Resolution,
) -> ProblemSpec:
    data = spec.model_dump()

    for update in resolution.updates:
        if not update.path:
            raise ValueError("Update path cannot be empty")

        current = data

        for part in update.path[:-1]:
            if isinstance(part, str):
                if not isinstance(current, dict) or part not in current:
                    raise KeyError(f"Invalid update path: {update.path}")
                current = current[part]

            elif isinstance(part, int):
                if not isinstance(current, list):
                    raise TypeError(f"Expected a list at: {update.path}")
                current = current[part]

        last = update.path[-1]

        if isinstance(last, str):
            if not isinstance(current, dict) or last not in current:
                raise KeyError(f"Invalid update path: {update.path}")
            current[last] = update.value

        elif isinstance(last, int):
            if not isinstance(current, list):
                raise TypeError(f"Expected a list at: {update.path}")
            current[last] = update.value

    return ProblemSpec.model_validate(data)


def deterministic_blockers(session: Session) -> list[str]:
    blockers = []

    if session.spec.problem_kind == "unknown":
        blockers.append("Problem type is still unknown.")

    if session.spec.physics is None:
        blockers.append("Physics has not been identified.")

    elif session.spec.physics.family == "unknown":
        blockers.append("Physics family is still unknown.")

    if session.spec.problem_kind == "optimization":
        if session.spec.optimization is None:
            blockers.append("Optimization information is missing.")

        elif not session.spec.optimization.objectives:
            blockers.append("Optimization objective is missing.")

    return blockers


def ready_to_try(session: Session) -> bool:
    if deterministic_blockers(session):
        return False

    return not any(
        issue.blocking
        for issue in session.review.issues
    )


def start_session(
    request: str,
    context: str | None = None,
) -> Session:
    spec, parse_usage = parse_problem(
        request,
        context,
    )

    review, review_usage = review_problem(spec)

    return Session(
        original_request=request,
        context=context,
        spec=spec,
        review=review,
        usage=[
            parse_usage,
            review_usage,
        ],
    )


def answer_questions(
    session: Session,
    answers: dict[str, str],
) -> Session:
    session = Session.model_validate(session.model_dump())

    resolution, resolve_usage = resolve_answers(
        session.spec,
        session.review,
        answers,
    )

    new_spec = apply_resolution(
        session.spec,
        resolution,
    )

    new_review, review_usage = review_problem(
        new_spec,
    )

    revision = Revision(
        answers=answers,
        resolution=resolution,
    )

    return Session(
        original_request=session.original_request,
        context=session.context,
        spec=new_spec,
        review=new_review,
        revisions=[
            *session.revisions,
            revision,
        ],
        usage=[
            *session.usage,
            resolve_usage,
            review_usage,
        ],
    )