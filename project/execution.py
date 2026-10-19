"""Reviewed specification -> isolated MPI process, with durable run evidence."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from project.models import Session
from project.storage import canonical, digest, utc_now, write_json

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
PVBATCH = Path("/opt/paraview/bin/pvbatch")
STARTUP_GRACE_SECONDS = 30
MAX_ITERATIONS = 500   # largest iteration budget an approval may grant
MAX_RANKS = 8          # most MPI processes a run may use
RENDERING_STEP = "rendering with ParaView"   # status step while pvbatch renders a finished run


class _ApprovalMismatch(ValueError):
    """Saved inputs have changed since the engineer approved the plan."""


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
        write_json(path, status)
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


def run_plan(session: Session, max_iterations: int, ranks: int) -> dict:
    from project.lbracket import assess_spec
    from project.workflow import ready_to_try

    if not ready_to_try(session):
        raise ValueError("Resolve blocking specification issues before running.")
    assessment = assess_spec(session.spec)
    if not assessment.ready or assessment.config is None:
        raise ValueError("The specification is not supported by the 2D L-bracket solver.")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or not 1 <= max_iterations <= MAX_ITERATIONS:
        raise ValueError(f"Iteration budget must be an integer between 1 and {MAX_ITERATIONS}.")
    if isinstance(ranks, bool) or not isinstance(ranks, int) or not 1 <= ranks <= MAX_RANKS:
        raise ValueError(f"MPI ranks must be an integer between 1 and {MAX_RANKS}.")
    plan = {
        "spec": session.spec.model_dump(mode="json"),
        "config": assessment.config.to_dict(),
        "max_iterations": max_iterations,
        "mpi_ranks": ranks,
    }
    plan["approval_hash"] = digest(plan)
    return plan


def launch_run(session: Session, *, max_iterations: int, ranks: int, approver: str) -> Path:
    plan = run_plan(session, max_iterations, ranks)
    if ranks > 1 and shutil.which("mpiexec") is None:
        raise RuntimeError("MPI is unavailable. Start the supplied Docker image.")
    run_dir = ARTIFACTS / "runs" / "lbracket2d" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:10]
    )
    run_dir.mkdir(parents=True)
    write_json(run_dir / "config.json", plan["config"])
    write_json(run_dir / "session.json", session.model_dump(mode="json"))
    write_json(run_dir / "approval.json", {
        **plan, "approved_at": utc_now(), "approver": approver,
        "scope": "Execute this configuration for the recorded iteration budget.",
    })
    _set_status(run_dir, state="starting", started_at=utc_now())
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
            write_json(run_dir / "worker.json", identity)
        elif worker.poll() is not None:
            # A very early import failure can prevent the worker from ever
            # writing its own receipt. Preserve a terminal receipt if it did.
            with _status_lock(run_dir):
                path = run_dir / "status.json"
                status = json.loads(path.read_text())
                if status["state"] in {"starting", "running"}:
                    write_json(path, {**status, "state": "failed", "reason": "Solver worker exited during startup.",
                                      "returncode": worker.returncode, "finished_at": utc_now()})
    except OSError:
        _set_status(run_dir, state="failed", reason="Unable to start solver worker.", finished_at=utc_now())
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
                  "finished_at": utc_now(), "convergence": "not assessed"}
        write_json(path, status)
        return status


def _render_paraview(run_dir: Path) -> dict:
    """Render the finished run's element densities with ParaView (pvbatch), as part of the run.

    ParaView comes from the host (mounted at /opt/paraview by docker-compose.yml). Without it
    the run still completes and records why no ParaView render exists.
    """
    if not PVBATCH.exists():
        return {"state": "unavailable", "reason": f"{PVBATCH} not found; mount a ParaView install at /opt/paraview."}
    _set_status(run_dir, step=RENDERING_STEP)
    script = Path(__file__).resolve().parent / "solver" / "lbracket2d" / "render_paraview.py"
    environment = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    try:
        with (run_dir / "paraview.log").open("w") as output:
            result = subprocess.run([str(PVBATCH), "--opengl-window-backend", "OSMesa", str(script),
                                     str(run_dir / "solver"), str(run_dir / "paraview")],
                                    stdout=output, stderr=subprocess.STDOUT, env=environment, timeout=900, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"state": "failed", "reason": type(error).__name__}
    if result.returncode != 0 or not (run_dir / "paraview" / "manifest.json").exists():
        return {"state": "failed", "reason": f"pvbatch exited with code {result.returncode}; see paraview.log"}
    frames = json.loads((run_dir / "paraview" / "manifest.json").read_text())["frames"]
    return {"state": "rendered", "frames": len(frames), "renderer": str(PVBATCH)}


def _worker(run_dir: Path) -> int:
    try:
        identity = _process_identity(os.getpid())
        if identity is None:
            raise RuntimeError("The supplied Linux Docker runtime is required to track solver workers.")
        write_json(run_dir / "worker.json", identity)
        approval = json.loads((run_dir / "approval.json").read_text())
        session = Session.model_validate_json((run_dir / "session.json").read_text())
        plan = run_plan(session, approval["max_iterations"], approval["mpi_ranks"])
        saved_config = json.loads((run_dir / "config.json").read_text())
        if (plan["approval_hash"] != approval["approval_hash"]
                or canonical(saved_config) != canonical(plan["config"])
                or canonical(approval["config"]) != canonical(plan["config"])
                or canonical(approval["spec"]) != canonical(plan["spec"])):
            raise _ApprovalMismatch()
        command = [sys.executable, "-m", "project.solver.lbracket2d.main",
                   "--config", str(run_dir / "config.json"), "--output", str(run_dir / "solver"),
                   "--max-iterations", str(plan["max_iterations"])]
        if plan["mpi_ranks"] > 1:
            command = ["mpiexec", "-n", str(plan["mpi_ranks"]), *command]
        _set_status(run_dir, state="running", command=command)
        with (run_dir / "solver.log").open("w") as output:
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=False)
        paraview = _render_paraview(run_dir) if result.returncode == 0 else {"state": "skipped", "reason": "solver failed"}
        state = "completed" if result.returncode == 0 else "failed"
        _set_status(run_dir, state=state, step=None, returncode=result.returncode, finished_at=utc_now(),
                    convergence="not assessed", paraview=paraview)
        return result.returncode
    except Exception as error:
        reason = ("Saved solver inputs no longer match the approved plan. Review and approve a new run."
                  if isinstance(error, _ApprovalMismatch) else type(error).__name__)
        _set_status(run_dir, state="failed", step=None, reason=reason, finished_at=utc_now(),
                    convergence="not assessed")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Solver worker for one approved run; launch_run starts it.")
    parser.add_argument("--worker", type=Path, required=True, help="Run directory prepared by launch_run.")
    args = parser.parse_args()
    return _worker(args.worker.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
