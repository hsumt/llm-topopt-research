"""
Written 7/23/26 
Records that the numerical verification tests passed for the exact current code version. Computes SHA-256 hashes for important source files and creats the manifest.  
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


VERIFIED_RELATIVE_FILES = [
    "agent/_critic.py",
    "agent/_physics.py",
    "parser/schema.py",
    "parser/prompt.py",
    "parser/client.py",
    "parser/provenance.py",
    "run_system.py",
    "batch_runner.py",
    "solver/SIMP_MASTER.py",
    "solver/_config_bridge.py",
    "solver/_verification_manifest.py",
    "solver/_02_FEA/_assembly.py",
    "solver/_02_FEA/_solver.py",
    "solver/_03_OPTMZER/_filters.py",
    "solver/_03_OPTMZER/_objective.py",
    "solver/_03_OPTMZER/_MMAupdate.py",
    "solver/reference_q4.py",
    "solver/mesh_refinement_study.py",
    "solver/diagnostic_calibration.py",
    "solver/verification_tests.py",
]

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(project_root: Path) -> dict[str, str]:
    hashes = {}
    for relative in VERIFIED_RELATIVE_FILES:
        path = project_root / relative
        if not path.exists():
            raise FileNotFoundError(f"Verification source file missing: {path}")
        hashes[relative] = _sha256(path)
    return hashes


def manifest_path(project_root: Path) -> Path:
    return project_root / "solver" / "_05_OUT" / "verification" / "verification_manifest.json"


def write_manifest(project_root: Path, *, tests: list[str], versions: dict) -> Path:
    path = manifest_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "passed": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "tests": list(tests),
        "versions": versions,
        "source_hashes": source_hashes(project_root),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_manifest_status(project_root: Path) -> dict:
    path = manifest_path(project_root)
    if not path.exists():
        return {
            "available": False,
            "current": False,
            "passed": False,
            "reason": "verification manifest not found; run verification_tests.py",
            "path": str(path),
        }
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "available": True,
            "current": False,
            "passed": False,
            "reason": f"verification manifest unreadable: {exc}",
            "path": str(path),
        }
    try:
        current_hashes = source_hashes(project_root)
    except (OSError, FileNotFoundError) as exc:
        return {
            "available": True,
            "current": False,
            "passed": False,
            "reason": str(exc),
            "path": str(path),
            "manifest": payload,
        }
    expected_hashes = payload.get("source_hashes", {})
    mismatches = sorted(
        relative
        for relative, current in current_hashes.items()
        if expected_hashes.get(relative) != current
    )
    passed = bool(payload.get("passed", False))
    current = not mismatches
    reason = "current verification manifest" if passed and current else (
        "verified source files changed after the suite ran: " + ", ".join(mismatches)
        if mismatches
        else "verification manifest does not report success"
    )
    return {
        "available": True,
        "current": current,
        "passed": passed and current,
        "reason": reason,
        "path": str(path),
        "mismatched_files": mismatches,
        "manifest": payload,
    }


def load_hash_bound_artifact(path: Path, project_root: Path, *, success_key: str) -> dict:
    """Validate a study/calibration manifest against current source hashes."""
    if not path.exists():
        return {
            "available": False,
            "current": False,
            "passed": False,
            "reason": f"manifest not found: {path}",
            "path": str(path),
        }
    try:
        payload = json.loads(path.read_text())
        current_hashes = source_hashes(project_root)
    except (OSError, json.JSONDecodeError, FileNotFoundError) as exc:
        return {
            "available": True,
            "current": False,
            "passed": False,
            "reason": f"manifest could not be validated: {exc}",
            "path": str(path),
        }
    expected = payload.get("source_hashes", {})
    mismatches = sorted(
        relative
        for relative, current in current_hashes.items()
        if expected.get(relative) != current
    )
    successful = bool(payload.get(success_key, False))
    current = not mismatches
    return {
        "available": True,
        "current": current,
        "passed": successful and current,
        "reason": (
            "current completed manifest"
            if successful and current
            else (
                "source files changed after the manifest was generated: "
                + ", ".join(mismatches)
                if mismatches
                else f"manifest does not set {success_key}=true"
            )
        ),
        "path": str(path),
        "mismatched_files": mismatches,
        "manifest": payload,
    }
