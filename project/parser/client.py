# client.py
from typing import Tuple, List

from dotenv import load_dotenv
import json
from anthropic import Anthropic
from project.parser.prompt import SYSTEM_PROMPT
from schema import ProblemSpec, ParserResult, DefaultedField
import os

load_dotenv()  # reads .env
api_key = os.environ["ANTHROPIC_API_KEY"]

client = Anthropic(api_key=api_key)


# client.py
from typing import Tuple, List

from dotenv import load_dotenv
import json
from anthropic import Anthropic
from project.parser.prompt import SYSTEM_PROMPT
from schema import ProblemSpec, ParserResult, DefaultedField
import os

load_dotenv()  # reads .env
api_key = os.environ["ANTHROPIC_API_KEY"]

client = Anthropic(api_key=api_key)


# ---------------------------------------------------------------------
# Deterministic r_min default. NOT computed by the LLM — this is the
# one number in the whole pipeline most likely to silently break a run
# if it's wrong (see: every failed run defaulting to r_min=1.5 against
# a ~0.05-wide element, producing a filter ~30 elements wide).
#
# Stays entirely in physical units, no unit-system conversion. Targets
# "filter spans ~2.5 elements" as the actual invariant, then clamps
# into the 0.01-0.10 physical-unit range already validated by hand
# across the working benchmark runs (cantilever: 0.05, MBB: 0.06).
# ---------------------------------------------------------------------
DEFAULT_R_MIN_FLOOR = 0.01
DEFAULT_R_MIN_CEIL  = 0.10
R_MIN_ELEMENT_MULTIPLIER = 2.5


def compute_default_r_min(Lx: float, Ly: float, nx: int, ny: int) -> float:
    element_size = min(Lx / nx, Ly / ny)
    r_min = R_MIN_ELEMENT_MULTIPLIER * element_size
    return max(DEFAULT_R_MIN_FLOOR, min(r_min, DEFAULT_R_MIN_CEIL))


def parse_problem(prompt: str) -> Tuple[ProblemSpec, List[DefaultedField], dict]:
    """
    Returns (spec, defaulted_fields, parser_tokens).

    spec is always a complete, runnable ProblemSpec. defaulted_fields is
    the list of fields the parser filled in rather than read from the
    prompt; the CALLER decides whether to surface a clarifying question.

    IMPORTANT: if simp.r_min appears in defaulted_fields, this function
    overrides the LLM's emitted value with compute_default_r_min() below,
    deterministically, in Python. The LLM's own arithmetic for this one
    field is never trusted, even though the prompt also asks it to
    compute a mesh-relative value — this is a deliberate belt-and-
    suspenders choice given how badly a wrong r_min breaks a run.
    """
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,  # raised again: a sparse prompt (e.g. "Optimize a
                          # cantilever beam") can default 10+ fields, each
                          # needing a full question string in defaulted_fields.
                          # 900 truncated mid-output on exactly this case.
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    print("Parser usage:")
    print("  input tokens :", response.usage.input_tokens)
    print("  output tokens:", response.usage.output_tokens)
    print("  total tokens :", response.usage.input_tokens + response.usage.output_tokens)

    parser_tokens = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "total_tokens": (
            response.usage.input_tokens +
            response.usage.output_tokens
        )
    }

    # Explicit truncation check. response.stop_reason == "max_tokens" means
    # the model was cut off mid-generation — surface THAT as the error,
    # not a confusing downstream JSONDecodeError pointing at an arbitrary
    # line/column where the truncated text happens to break parsing.
    if response.stop_reason == "max_tokens":
        raise ValueError(
            f"Parser response was truncated at max_tokens "
            f"({response.usage.output_tokens} output tokens used). "
            f"The prompt likely left many fields unstated, producing a "
            f"long defaulted_fields list. Raise max_tokens in client.py "
            f"or shorten the question text in prompt.py."
        )

    text = response.content[0].text.strip()
    start_idx = text.find('{')
    end_idx = text.rfind('}') + 1
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON:\n{text}") from e

    try:
        result = ParserResult(**data)
    except Exception as e:
        raise ValueError(f"JSON does not match ParserResult schema: {data}") from e

    spec = result.spec
    defaulted_fields = result.defaulted_fields

    # Override r_min deterministically if it was defaulted.
    r_min_was_defaulted = any(
        f.field_path == "simp.r_min" for f in defaulted_fields
    )
    if r_min_was_defaulted:
        Lx = spec.mesh.Lx if spec.mesh.Lx is not None else spec.mesh.nx / spec.mesh.ny
        Ly = spec.mesh.Ly if spec.mesh.Ly is not None else 1.0
        correct_r_min = compute_default_r_min(Lx, Ly, spec.mesh.nx, spec.mesh.ny)

        spec_data = spec.model_dump()
        spec_data["simp"]["r_min"] = correct_r_min
        spec = ProblemSpec(**spec_data)

        # Keep defaulted_fields' reported value honest — it should match
        # what actually ended up in the spec, not the LLM's original guess.
        for f in defaulted_fields:
            if f.field_path == "simp.r_min":
                f.default_used = correct_r_min
                f.question = (
                    f"What filter radius should I use? I'll default to "
                    f"{correct_r_min:.4f} (~2.5 elements wide for this "
                    f"mesh) if you're not sure."
                )

    return spec, defaulted_fields, parser_tokens