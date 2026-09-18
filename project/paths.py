"""Canonical repository paths for source code and generated artifacts."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_ROOT.parent

ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts"
RUNS_ROOT = ARTIFACT_ROOT / "runs"
FORMULATION_SESSIONS_ROOT = ARTIFACT_ROOT / "formulation_sessions"
FORMULATION_EVALUATION_ROOT = ARTIFACT_ROOT / "formulation_evaluation"
FORMULATION_TRAINING_ROOT = ARTIFACT_ROOT / "formulation_training"
VERIFICATION_ARTIFACT_ROOT = ARTIFACT_ROOT / "verification"
VV_ARTIFACT_ROOT = ARTIFACT_ROOT / "vv"