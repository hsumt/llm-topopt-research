from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

class Model(BaseModel): # throws errors if extra comes in
    model_config = ConfigDict(extra="forbid")
class Value(Model): # any value with units (125 N for example)
    value: Any
    unit: str | None = None
class Region(Model): # regions of the model
    name: str
    description: str
    role: str | None = None
class Geometry(Model): 
    physical_dimension: Literal[1, 2, 3] | None = None
    description: str | None = None
    parameters: dict[str,Value] = Field(default_factory=dict)
    regions: list[Region] = Field(default_factory=list)

class Material(Model):
    name: str | None = None
    region: str | None = None
    properties: dict[str, Value] = Field(default_factory=dict)

class BoundaryCondition(Model):
    name: str
    location: str | None = None
    kind: str | None = None
    components: list[str] = Field(default_factory=list)
    value: Value | None = None
class Load(Model):
    name: str
    location: str | None = None
    kind: str | None = None
    magnitude: Value | None = None
    direction: str | None = None
    load_case: str | None = None
class Objective(Model):
    sense: Literal["minimize", "maximize"] | None = None
    quantity: str | None = None
    aggregation: str | None = None


class Constraint(Model):
    quantity: str | None = None
    relation: Literal["<=", ">=", "="] | None = None
    limit: Value | None = None
    region: str | None = None


class Optimization(Model):
    design_variable: str | None = None
    objectives: list[Objective] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)


class Manufacturing(Model):
    process: str | None = None
    requirements: list[str] = Field(default_factory=list)

class Physics(Model):
    family: Literal[ # honestly, names have been written out of braindumping.
        "solid_mechanics",
        "thermal",
        "fluid",
        "electromagnetics",
        "other",
        "quantum",
    ] = "multiphysics"

    representation: str | None = None
    regime: str | None = None
class ProblemSpec(Model):
    name: str
    problem_kind: Literal[
        "simulation",
        "optimization",
        "inverse_problem",
        "unknown",
    ] = "unknown"

    geometry: Geometry | None = None
    physics: Physics | None = None

    materials: list[Material] = Field(default_factory=list)
    boundary_conditions: list[BoundaryCondition] = Field(default_factory=list)
    loads: list[Load] = Field(default_factory=list)

    optimization: Optimization | None = None
    manufacturing: Manufacturing | None = None

    assumptions: list[str] = Field(default_factory=list)

class Issue(Model): #issue in the parsing
    key: str
    field: str | None = None
    description: str
    blocking: bool = True


class Question(Model): # clarifying question
    key: str
    issue_keys: list[str] = Field(default_factory=list)
    prompt: str
    why: str
    answer_type: Literal["text", "single_choice"] = "text"
    options: list[str] = Field(default_factory=list)


class Review(Model): #part of the reviews. Lists issues and questions
    issues: list[Issue] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)

class Update(Model): # updates the fields with answers
    path: list[str | int]
    value: Any


class Resolution(Model):
    updates: list[Update] = Field(default_factory=list)
    resolved_issue_keys: list[str] = Field(default_factory=list)


class Usage(Model): # usage for tokens we can get tokens by cost = n_input x p_input + n_output x p_output
    step: Literal["parse", "review", "resolve", "vision"]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit: bool = False


class Revision(Model): # what was asked -> what the human said -> how the AI interpreted that answer -> what changed in the spec
    answers: dict[str, str] = Field(default_factory=dict)
    resolution: Resolution


class Session(Model): # combines the original request + specification + review (questions/issues) + revisions + AI usage
    original_request: str
    context: str | None = None
    spec: ProblemSpec
    review: Review
    revisions: list[Revision] = Field(default_factory=list)
    usage: list[Usage] = Field(default_factory=list)