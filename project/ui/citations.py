"""Code citations for the process flow: each claim points at the code that implements it.

An anchor is either "module:qualname" (that function's or class's source) or "path#text"
(the line containing that text, with a few lines either side). Anchors resolve against the
live source, so the line numbers stay right when the code moves. A rule id resolves to the
line that applies it.
"""
from functools import lru_cache
import importlib
import inspect
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # repository root (/app in the container)
MAX_LINES = 60
CONTEXT = 4
_RULE_FILES = {"assess": ("project/lbracket.py",), "merge": ("project/lbracket.py",),
               "ui": ("project/ui/api.py",), "apply": ("project/workflow.py", "project/trace.py")}
_DICT_ENTRY = re.compile(r'^\s*"[\w.]+":\s*[("]')   # a description table entry, not the code that applies a rule


def _entry(path: str, lines: list[str], first: int, start: int, end: int, truncated: bool = False) -> dict:
    host = os.environ.get("ATO_HOST_REPO")
    return {"file": path, "first": first, "start": start, "end": end, "lines": lines, "truncated": truncated,
            "label": f"{Path(path).name}:{start}" + (f"-{end}" if end != start else ""),
            "open": f"vscode://file/{host.rstrip('/')}/{path}:{start}" if host else None}


def _around(path: str, index: int) -> dict:
    lines = (ROOT / path).read_text().splitlines()
    start, end = max(0, index - CONTEXT), min(len(lines), index + CONTEXT + 1)
    return _entry(path, lines[start:end], start + 1, index + 1, index + 1)


@lru_cache(maxsize=512)
def cite(anchor: str) -> dict | None:
    """Resolve an anchor to {file, start, end, lines, label, open}, or None if it no longer exists."""
    try:
        if "#" in anchor:
            path, text = anchor.split("#", 1)
            lines = (ROOT / path).read_text().splitlines()
            hits = [i for i, line in enumerate(lines) if text in line]
            index = next((i for i in hits if not _DICT_ENTRY.match(lines[i])), hits[0])
            return _around(path, index)
        module, _, qualname = anchor.partition(":")
        target = importlib.import_module(module)
        for part in qualname.split("."):
            target = getattr(target, part)
        source, start = inspect.getsourcelines(target)
        path = Path(inspect.getsourcefile(target)).resolve().relative_to(ROOT).as_posix()
        return _entry(path, [line.rstrip("\n") for line in source[:MAX_LINES]], start, start,
                      start + len(source) - 1, truncated=len(source) > MAX_LINES)
    except Exception:
        return None


@lru_cache(maxsize=512)
def rule(rule_id: str | None) -> dict | None:
    """The line that applies a rule id such as merge.cap_4 or assess.support@lbracket.py:244."""
    if not rule_id:
        return None
    try:
        if "@" in rule_id:
            # assess.<key>@lbracket.py:<line>: the line was recorded when the trace was written; if
            # the code has moved since, use the add("<key>", ...) call nearest to it.
            name, location = rule_id.split("@", 1)
            file, line = location.rsplit(":", 1)
            path, recorded = f"project/{file}", int(line) - 1
            lines = (ROOT / path).read_text().splitlines()
            call = f'add("{name.split(".", 1)[1]}"'
            if not (0 <= recorded < len(lines) and call in lines[recorded]):
                hits = [i for i, text in enumerate(lines) if call in text]
                recorded = min(hits, key=lambda i: abs(i - recorded)) if hits else recorded
            return _around(path, recorded)
        for path in _RULE_FILES.get(rule_id.split(".", 1)[0], ()):
            found = cite(f'{path}#"{rule_id}"')
            if found:
                return found
        return None
    except Exception:
        return None
