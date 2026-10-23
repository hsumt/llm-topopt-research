"""Role-coded view of a problem specification, plus the value formatting other components share.

Each value is tagged with what the connected solver makes of it (ui/spec_model.py builds the rows,
spec_card.py draws them).
"""
import re

from project.ui import api

_MARKDOWN = re.compile(r"([\\`*_\[\]<>#$~|])")


def md(text) -> str:
    """Escape model-written text so st.markdown shows it literally. A zero-width space after each ':' and
    in "www." keeps Streamlit directives (:red[..]), icons (:material/..:) and links from forming."""
    escaped = _MARKDOWN.sub(r"\\\1", str(text)).replace(":", ":\u200b")
    return re.sub(r"(?i)\bwww\.", lambda found: found.group(0)[:3] + "\u200b.", escaped)


def words(text) -> str | None:
    """Plain words for an identifier; None when unset or unknown."""
    if text is None or str(text).strip() in ("", "unknown"):
        return None
    return str(text).replace("_", " ").strip()


def capital(text: str | None) -> str | None:
    return text[:1].upper() + text[1:] if text else text


def number(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(number(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {number(v)}" for k, v in value.items()) + "}"
    return str(value)


def _numbers(value) -> list[float] | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return [abs(value)]
    if isinstance(value, (list, tuple)):
        found: list[float] = []
        for item in value:
            inner = _numbers(item)
            if inner is None:
                return None
            found += inner
        return found
    return None


def _scale(value, factor: float):
    return [_scale(v, factor) for v in value] if isinstance(value, (list, tuple)) else value * factor


def readable(value, unit: str | None) -> tuple[str, str]:
    """(value, unit) as display text: metres as mm, pascals as MPa or GPa."""
    numbers = _numbers(value)
    if numbers and unit == "m":
        value, unit = _scale(value, 1e3), "mm"
    elif numbers and unit == "Pa":
        biggest = max(numbers)
        if biggest >= 1e9:
            value, unit = _scale(value, 1e-9), "GPa"
        elif biggest >= 1e6:
            value, unit = _scale(value, 1e-6), "MPa"
    return number(value), unit or ""


def quantity(item: dict | None) -> str | None:
    if not item or item.get("value") is None:
        return None
    return " ".join(part for part in readable(item["value"], item.get("unit")) if part)


def _dump(spec) -> dict:
    return spec.model_dump(mode="json") if hasattr(spec, "model_dump") else dict(spec or {})


def render(spec, *, key: str, kept: frozenset[str] = frozenset()) -> None:
    """Specification card of a ProblemSpec (model or dict): one card per section, each value with its role.
    key: unique on the page (a batch can show the same spec twice). kept: the session's kept assumptions
    (workflow.kept_assumptions), for the current spec of a formulation."""
    # Imported here: spec_model imports the helpers above, and they stay importable without streamlit.
    from project.ui import spec_model
    from project.ui.components import spec_card
    data = _dump(spec)
    spec_card.render(spec_model.build(data, api.spec_roles(data, kept)), key=key)
