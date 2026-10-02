"""Reviewed specification -> isolated MPI process, with durable run evidence."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from project.models import Session

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
STARTUP_GRACE_SECONDS = 30


class _ApprovalMismatch(ValueError):
    """Saved inputs have changed since the engineer approved the plan."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(data) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _status_lock(run_dir: Path):
    """Serialize worker completion with readers reconciling interrupted runs."""
    with (run_dir / "status.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _set_status(run_dir: Path, **updates) -> dict:
    with _status_lock(run_dir):
        path = run_dir / "status.json"
        previous = json.loads(path.read_text()) if path.exists() else {}
        status = {**previous, **updates}
        _write(path, status)
        return status


def _process_identity(pid: int) -> dict | None:
    """Identify one live Linux process, including across container/PID reuse.

    An exited but unreaped process is not live. No process is signalled when
    inspecting a saved run, including when its old PID has been reused.
    """
    try:
        proc = Path("/proc") / str(pid)
        # comm (field 2) can contain spaces and parentheses; fields after its
        # final closing parenthesis begin with state (3), starttime is 22.
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in {"Z", "X", "x"}:
            return None
        return {
            "pid": pid,
            "start_ticks": int(fields[19]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "pid_namespace": os.readlink(proc / "ns/pid"),
        }
    except (OSError, ValueError, IndexError):
        return None


def run_plan(session: Session, max_iterations: int = 2, ranks: int = 1) -> dict:
    from project.lbracket import assess_spec
    from project.workflow import ready_to_try

    if not ready_to_try(session):
        raise ValueError("Resolve blocking specification issues before running.")
    assessment = assess_spec(session.spec)
    if not assessment.ready or assessment.config is None:
        raise ValueError("The specification is not supported by the 3D L-bracket solver.")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or not 1 <= max_iterations <= 500:
        raise ValueError("Iteration budget must be an integer between 1 and 500.")
    if isinstance(ranks, bool) or not isinstance(ranks, int) or not 1 <= ranks <= 8:
        raise ValueError("MPI ranks must be an integer between 1 and 8.")
    plan = {
        "spec": session.spec.model_dump(mode="json"),
        "config": assessment.config.to_dict(),
        "max_iterations": max_iterations,
        "mpi_ranks": ranks,
    }
    encoded = _canonical(plan)
    plan["approval_hash"] = hashlib.sha256(encoded.encode()).hexdigest()
    return plan


def launch_run(session: Session, *, approved_hash: str, max_iterations: int = 2,
               ranks: int = 1, approver: str = "local engineer") -> Path:
    plan = run_plan(session, max_iterations, ranks)
    if approved_hash != plan["approval_hash"]:
        raise ValueError("The specification or run settings changed; review the current configuration.")
    if ranks > 1 and shutil.which("mpiexec") is None:
        raise RuntimeError("MPI is unavailable. Start the supplied Docker image.")
    run_dir = ARTIFACTS / "runs" / "lbracket3d" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:10]
    )
    run_dir.mkdir(parents=True)
    _write(run_dir / "config.json", plan["config"])
    _write(run_dir / "session.json", session.model_dump(mode="json"))
    _write(run_dir / "approval.json", {
        **plan, "approved_at": _now(), "approver": approver,
        "scope": "Execute this configuration for the recorded iteration budget.",
    })
    _set_status(run_dir, state="starting", started_at=_now())
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    # Numerical workers do not need access to the model credential.
    environment.pop("ANTHROPIC_API_KEY", None)
    try:
        with (run_dir / "worker.log").open("w") as output:
            worker = subprocess.Popen(
                [sys.executable, "-m", "project.execution", "--worker", str(run_dir)],
                cwd=Path(__file__).resolve().parents[1], env=environment,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
            )
        identity = _process_identity(worker.pid)
        if identity is not None:
            _write(run_dir / "worker.json", identity)
        elif worker.poll() is not None:
            # A very early import failure can prevent the worker from ever
            # writing its own receipt. Preserve a terminal receipt if it did.
            with _status_lock(run_dir):
                path = run_dir / "status.json"
                status = json.loads(path.read_text())
                if status["state"] in {"starting", "running"}:
                    _write(path, {**status, "state": "failed", "reason": "Solver worker exited during startup.",
                                  "returncode": worker.returncode, "finished_at": _now()})
    except OSError:
        _set_status(run_dir, state="failed", reason="Unable to start solver worker.", finished_at=_now())
        raise
    return run_dir


def run_status(run_dir: Path) -> dict:
    with _status_lock(run_dir):
        path = run_dir / "status.json"
        status = json.loads(path.read_text())
        if status["state"] not in {"starting", "running"}:
            return status
        worker_path = run_dir / "worker.json"
        if worker_path.exists():
            identity = json.loads(worker_path.read_text())
            if _process_identity(identity["pid"]) == identity:
                return status
            reason = "Solver worker stopped before recording completion. Inspect the solver log and outputs."
        else:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(status["started_at"])).total_seconds()
            if age < STARTUP_GRACE_SECONDS:
                return status
            reason = "Solver worker did not register its process during startup. Inspect the worker log."
        status = {**status, "state": "interrupted", "reason": reason,
                  "finished_at": _now(), "convergence": "not assessed"}
        _write(path, status)
        return status


def _worker(run_dir: Path) -> int:
    try:
        identity = _process_identity(os.getpid())
        if identity is None:
            raise RuntimeError("The supplied Linux Docker runtime is required to track solver workers.")
        _write(run_dir / "worker.json", identity)
        approval = json.loads((run_dir / "approval.json").read_text())
        session = Session.model_validate_json((run_dir / "session.json").read_text())
        plan = run_plan(session, approval["max_iterations"], approval["mpi_ranks"])
        saved_config = json.loads((run_dir / "config.json").read_text())
        if (plan["approval_hash"] != approval["approval_hash"]
                or _canonical(saved_config) != _canonical(plan["config"])
                or _canonical(approval["config"]) != _canonical(plan["config"])
                or _canonical(approval["spec"]) != _canonical(plan["spec"])):
            raise _ApprovalMismatch()
        command = [sys.executable, "-m", "project.solver.lbracket3d.main",
                   "--config", str(run_dir / "config.json"), "--output", str(run_dir / "solver"),
                   "--max-iterations", str(plan["max_iterations"])]
        if plan["mpi_ranks"] > 1:
            command = ["mpiexec", "-n", str(plan["mpi_ranks"]), *command]
        _set_status(run_dir, state="running", command=command)
        with (run_dir / "solver.log").open("w") as output:
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=False)
        state = "completed" if result.returncode == 0 else "failed"
        _set_status(run_dir, state=state, returncode=result.returncode, finished_at=_now(),
                    convergence="not assessed")
        return result.returncode
    except Exception as error:
        reason = ("Saved solver inputs no longer match the approved plan. Review and approve a new run."
                  if isinstance(error, _ApprovalMismatch) else type(error).__name__)
        _set_status(run_dir, state="failed", reason=reason, finished_at=_now(),
                    convergence="not assessed")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the reviewed 3D L-bracket in the supplied Docker environment.")
    parser.add_argument("--reference", action="store_true", help="Explicitly select every physical input in the recorded five-hole benchmark.")
    parser.add_argument("--session", type=Path, help="Validated saved formulation Session JSON.")
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return _worker(args.worker.resolve())
    if bool(args.reference) == bool(args.session):
        parser.error("Choose exactly one of --reference or --session.")
    if args.reference:
        from project.lbracket import reference_spec
        from project.models import Review
        session = Session(original_request="Run the documented simplified_3D_holes benchmark.", spec=reference_spec(), review=Review())
    else:
        session = Session.model_validate_json(args.session.read_text())
    plan = run_plan(session, args.max_iterations, args.ranks)
    run_dir = launch_run(session, approved_hash=plan["approval_hash"], max_iterations=args.max_iterations,
                         ranks=args.ranks, approver="explicit CLI invocation")
    print(f"Run directory: {run_dir}", flush=True)
    import time
    while (status := run_status(run_dir))["state"] in {"starting", "running"}:
        time.sleep(0.5)
    print(json.dumps(status, indent=2))
    return 0 if status["state"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
