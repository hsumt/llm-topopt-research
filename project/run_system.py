import sys

sys.path.insert(0, "project/parser")
sys.path.insert(0, "project/solver")


from client import parse_problem
from SIMP_MASTER import main_from_spec

OPT_OUT_PHRASES = {"use defaults", "use default", "skip", "just use defaults"}


def _apply_overrides(spec, overrides: dict):
    """Applies user-provided overrides onto dotted field_path entries."""
    data = spec.model_dump()
    for path, value in overrides.items():
        parts = path.split(".")
        node = data
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = value
    return type(spec)(**data)


    
if __name__ == "__main__":
    prompt = input("Describe your topology optimization problem:\n> ")
 
    spec, defaulted_fields, parser_usage = parse_problem(prompt)
 
    if defaulted_fields:
        print(f"\n{len(defaulted_fields)} field(s) were not stated explicitly "
              f"and were defaulted. Press enter to accept a default, type a "
              f"number to override it, or type 'use defaults' to accept all "
              f"remaining defaults at once.\n")
 
        opted_out = False
        overrides = {}
 
        for field in defaulted_fields:
            if opted_out:
                print(f"  Using default for {field.field_path}: {field.default_used}")
                continue
 
            answer = input(f"  {field.question}\n  > ").strip()
 
            if answer.lower() in OPT_OUT_PHRASES:
                opted_out = True
                print("  Using defaults for all remaining fields.\n")
                continue
 
            if answer == "":
                continue  # accept the stated default for this field
 
            try:
                overrides[field.field_path] = float(answer)
            except ValueError:
                print(f"  Could not parse '{answer}' as a number — "
                      f"keeping default {field.default_used}.")
 
        if overrides:
            spec = _apply_overrides(spec, overrides)
 
    print("\nParsed spec:")
    print(spec.model_dump_json(indent=2))
    print("\nStarting SIMP optimization...\n")
    main_from_spec(spec, parser_usage=parser_usage)