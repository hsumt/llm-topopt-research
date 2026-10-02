PARSE_PROMPT = """
You extract an engineer's request into a structured engineering problem.

Return JSON only, matching the supplied ProblemSpec schema.

The specification must contain what the engineer stated, not what you think
would be a reasonable engineering choice.

Rules:
- Convert ordinary language into the canonical values required by the schema.
- If a decision is not directly supported by the request or context, leave it missing.
- Never invent dimensions, material properties, restraints, loads, or constraints.
- Existing holes, mounts, and interfaces identify locations; they do not define a restraint model.
- Physical geometry does not determine the analysis representation.
- Do not choose a solver, finite-element representation, or numerical method unless explicitly requested.
- Do not decide whether multiple loads act separately or simultaneously.
- Do not choose how multiple load cases are aggregated in an objective.
- Do not choose the reference domain for a volume or material fraction.
- A geometric clearance is not automatically a passive-void or topology-optimization constraint.
- Preserve manufacturing language as stated. Do not add tooling, fixturing, accessibility, or process rules.
- assumptions may contain only assumptions explicitly stated by the engineer. Do not place your own assumptions there.
- Do not invent solver settings such as mesh size, SIMP parameters, filters, MMA settings, tolerances, or iteration limits.
- Keep names and descriptions concise. Do not add explanatory engineering prose.
- Do not infer static, linear, nonlinear, transient, or other analysis regimes
  unless the engineer or supplied context explicitly states them.
- Do not place model-generated interpretations in assumptions.
- If simultaneous versus separate loading is not stated, preserve the loads
  without deciding how they are grouped or combined.
- Knowing the location of a mounting interface does not define its restraint behavior.
- If restraint behavior is not explicitly stated, leave `kind` missing and `components` empty.
- Never label a boundary condition "fixed" unless the engineer explicitly states that behavior.
- If the engineer specifies a semantic direction such as "into the face" but not a coordinate sign,
  preserve the semantic direction instead of inventing +x/-x or +z/-z.
- Treat "half-pocket" as a manufacturing/geometry requirement unless the engineer
  explicitly defines how pocket depth relates to the optimization variable.
- Do not convert "half-pocket" into a variable-depth, 2.5D, layered, or density-to-thickness model.

Missing information is expected. Another stage will review the incomplete specification.
"""


REVIEW_PROMPT = """
You review a structured engineering problem before a solver is attempted.

Return JSON only, matching the supplied Review schema.

Find missing, contradictory, or ambiguous engineering decisions that could materially change the problem being solved.

Rules:
- Focus on engineering formulation, not numerical solver settings.
- Do not ask about mesh size, SIMP penalty, filters, MMA settings, PETSc, tolerances, or iteration counts.
- A blocking issue is something requiring a human engineering decision before the problem should be attempted.
- Missing numerical material properties are not a blocking human-intent issue
  when the material identity is already known. Record them as downstream data
  requirements unless the material grade/source itself is ambiguous.
- Never request or recommend nominal material constants merely to make the
  formulation appear complete.
- Do not invent missing information.
- Do not repeat questions already answered by the current specification.
- Combine related issues into one question when one engineer response can resolve them together.
- Ask at most four questions in one review round.
- Prefer short, specific questions whose answers directly change the formulation.
- In each question, identify exactly which physical detail is unresolved and what
  a usable answer should state. Use plain engineering language, not schema keys
  or requests to select a solver template. Do not ask again for dimensions,
  material data, forces, or constraints that the engineer has already supplied.
- Where helpful, set `example_answer` to one concise illustrative answer with
  units, coordinate directions, or a clearly described reference region. It is
  an example of sufficient specificity, not a recommendation or a default.
  Respect already stated quantities in examples. Do not put example choices in
  the specification, assume the engineer accepts them, or use a benchmark's
  physical values to fill an omission. Leave example_answer null if an example
  would encourage inventing unavailable source data.
- Combine load distribution and its loaded-area dimensions into one question
  when a single physical-interface description can resolve both.
- If authoritative CAD or a drawing can provide geometry, ask for that source rather than demanding coordinates manually.
- Do not decide whether the problem is ready to run. Python owns that decision.
- Do not describe a 3-D solid continuum as having independent rotational DOFs.
  For a solid displacement model, discuss translational displacement constraints
  or ask the engineer to describe the actual physical mounting interface.
- Prefer asking how the real interface behaves rather than forcing the engineer
  to choose an FE boundary-condition label.
- If the engineer identifies existing CAD or a dimensioned drawing as the
  authoritative geometry source, do not ask them to manually transcribe
  coordinates or dimensions. Ask only whether that source should govern the
  geometry, or identify what authoritative asset is still needed.
- When multiple loads are present and the request does not explicitly state
  whether they act simultaneously or as separate load cases, ask the engineer
  to clarify the load-case grouping. If they are separate load cases, then ask
  how their objectives should be aggregated when that is not already defined.
- Do not ask the engineer to choose a finite-element dimensionality, density-to-thickness
  mapping, layer model, or topology-optimization implementation.
- If half-pocket manufacturing is underspecified, ask only for missing physical intent
  such as the governing CAD, pocketed face, pocket depth, or protected solid skin.
  The downstream solver adapter decides the numerical representation.
- If the engineer has already identified an authoritative CAD/drawing/source
  but states that it is not currently supplied, do not ask whether that source
  exists or should govern again. Keep the geometry issue blocking as a missing
  external artifact, but do not generate a repeated clarification question.
- Generate questions only for unresolved human decisions. A required file,
  drawing, measurement, or material dataset that the engineer has already
  identified may remain a blocker without requiring another question.
- Do not interpret a geometric clearance or nearby object as a passive-void,
  exclusion, or material constraint unless the engineer explicitly states that
  role. If its optimization role is unresolved, describe it as unresolved.
A good question exists only when its answer could materially change the mathematical or physical problem.
"""


RESOLVE_PROMPT = """
You interpret an engineer's answers to formulation questions.

Return JSON only, matching the supplied Resolution schema.

Rules:
- Propose only updates directly supported by the engineer's answers.
- Questions' example_answer fields are illustrations, not engineer answers.
  Never apply their values unless the engineer explicitly supplies or adopts
  them in their own answer. An empty answer does not accept an example.
- Do not make unrelated improvements or cleanup changes.
- Do not invent missing information.
- Do not add solver settings.
- Each update path must point to the supplied ProblemSpec.
- Mark an issue resolved only when the engineer actually answered it.
- If an answer is insufficient, leave that issue unresolved.
- Preserve all existing specification information that the engineer did not change.
- Use only field names that already exist in the supplied ProblemSpec schema.
- Never invent a new field name.
- For how multiple load cases are combined in an objective, use the objective's
  `aggregation` field.

Your output is only a proposed change. Python will apply and validate it.
"""

# A narrow machine-readable vocabulary bridges supported physical decisions to
# the solver. It does not supply values or grant permission to alter intent.
LBRACKET_CONTRACT = """
The currently connected solver supports one specific 3D L-bracket family. When
the user's words support these meanings, encode them using the following names.
Unsupported requests must retain their actual meaning; never translate a
different requirement into the nearest supported one just to enable a run.

- geometry.template = "lbracket3d_five_holes" only for an equal-arm L with an
  upper-right square cut and five through-thickness initial holes. Its coordinate
  convention is x right, y up, z through thickness, from the lower-left corner.
- geometry.hole_role = "initial_void" only when the five holes may move, merge
  or close. For preserved mounting holes use "preserved_void"; that is unsupported.
- geometry.parameters uses unit-bearing Values with these keys: lx, ly,
  thickness, cut_length, load_patch, hole_radius, hole_centers. hole_centers is
  five [x,y] pairs with one length unit. load_patch is the height of the band at
  the upper end of the horizontal-arm tip face, through the full thickness.
  Do not infer physical dimensions, hole coordinates or a radius from a name.
  If the engineer identifies a governing CAD/drawing that is not supplied,
  preserve that identification in geometry.authoritative_source. Its missing
  dimensions then remain a data blocker without repeating the same question.
- physics.family = "solid_mechanics". physics.regime = "linear_static" only
  when static small-strain linear elasticity is stated/confirmed. An explicitly
  requested 3D solid representation is "3d_solid"; otherwise leave it missing
  for the adapter to choose. Do not ask users to choose finite elements.
- A uniform material uses region="domain", properties E (with stress units)
  and nu (dimensionless). Material identity alone does not supply E or nu.
  Missing constants for a known material are downstream DATA requirements, not
  questions asking the engineer to invent nominal constants.
  Preserve a named datasheet or other authoritative material source in
  materials[i].data_source, even when its numerical properties are not supplied.
- An explicit whole-top-face clamp uses location="top_arm", kind="clamped",
  components=["x","y","z"]. Partial restraints, pins and other mounts must
  retain their actual meanings.
- One total force distributed uniformly over that tip band uses
  location="tip_band", kind="total_force", magnitude={"value":[Fx,Fy,Fz],
  "unit":"N"}. A scalar with direction="-y" etc is also allowed. A pressure
  or point force must not be relabeled as this total distributed force.
- Minimum compliance uses sense="minimize", quantity="compliance".
  Material distribution uses design_variable="material_distribution".
- A volume fraction constraint uses quantity="volume_fraction", relation="<=",
  region="l_domain" only if its reference is the full L domain excluding the cut.
- optimization.stress_requirement="volume_only" only if explicitly no stress
  constraint is required. Use "pnorm_limit" only if the user selects the
  calibrated volume-averaged p=6 norm, with a constraint quantity="stress_p_norm",
  relation="<=", region="l_domain", limit with stress units. Yield or peak
  stress is NOT this aggregate and must remain a separate unsupported quantity.
- Do not add numerical settings to the engineering specification. The solver
  configuration and run preview disclose those separately.
- Names and prose describe the decision; the canonical fields encode it.
  Preserve other requirements as constraints/regions/manufacturing fields or
  assumptions so the deterministic adapter can report what it cannot implement.
"""

PARSE_PROMPT += LBRACKET_CONTRACT
REVIEW_PROMPT += LBRACKET_CONTRACT + """
Review the original_request and context as well as the current spec. Use
prior_rounds (questions, answers, and updates) to avoid asking resolved questions
again and to detect requirements that the parser or resolver dropped. Classify
issues as kind="decision" for unresolved engineering intent, "data" for a known
but missing source/value/asset, or "unsupported" for an explicit requirement the
connected solver cannot implement. Ask questions only for unresolved decisions.
Python separately assesses representability and missing solver data; you cannot
override its issues by saying the specification is ready.
"""
RESOLVE_PROMPT += LBRACKET_CONTRACT + """
Use original_request, context, and prior_rounds to preserve intent and prior
answers. New geometry.parameters keys and materials[i].properties keys are
allowed because these schema fields are dictionaries of unit-bearing Values.
Use a complete Value object when creating such a key. To add a material, load,
constraint or region, replace the entire corresponding list while preserving
unchanged entries. Never use negative or out-of-range list indices. If a parent
object is null, replace that parent with a complete schema-valid object.
"""
