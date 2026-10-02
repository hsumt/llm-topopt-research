"""Execution approval and interruption checks; no numerical solve or API calls."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from project import execution
from project.lbracket import reference_spec
from project.models import Issue, Review, Session


class ExecutionTests(unittest.TestCase):
    def session(self):
        return Session(original_request="Use the documented benchmark",
                       spec=reference_spec(), review=Review())

    def test_approval_does_not_survive_changed_spec_or_budget(self):
        session = self.session()
        plan = execution.run_plan(session, 2, 1)
        with patch.object(execution.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(ValueError, "changed"):
                execution.launch_run(session, approved_hash=plan["approval_hash"], max_iterations=3)
            session.spec.loads[0].magnitude.value = [0, -4000, 0]
            session.spec.loads[0].direction = "vector"
            with self.assertRaisesRegex(ValueError, "changed"):
                execution.launch_run(session, approved_hash=plan["approval_hash"])
            spawn.assert_not_called()

    def test_blockers_cannot_be_bypassed_by_quiet_critic_or_approval(self):
        session = self.session()
        session.spec.loads = []
        with self.assertRaisesRegex(ValueError, "blocking"):
            execution.run_plan(session)
        session = self.session()
        session.review.issues = [Issue(key="uncertain_intent", description="Confirm the fixture")]
        with self.assertRaisesRegex(ValueError, "blocking"):
            execution.run_plan(session)

    def test_current_process_and_reused_pid_have_distinct_status(self):
        identity = execution._process_identity(os.getpid())
        self.assertIsNotNone(identity)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            execution._set_status(run, state="running", started_at=execution._now())
            execution._write(run / "worker.json", identity)
            self.assertEqual(execution.run_status(run)["state"], "running")
            execution._write(run / "worker.json", {**identity, "start_ticks": identity["start_ticks"] - 1})
            status = execution.run_status(run)
            self.assertEqual(status["state"], "interrupted")
            self.assertEqual(status["convergence"], "not assessed")
            self.assertEqual(json.loads((run / "status.json").read_text())["state"], "interrupted")

    def test_exited_child_is_not_running(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            execution._set_status(run, state="starting", started_at=execution._now())
            # The child exits only after we have recorded its Linux identity.
            child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)
            try:
                identity = execution._process_identity(child.pid)
                self.assertIsNotNone(identity)
                execution._write(run / "worker.json", identity)
                child.communicate(timeout=5)
                self.assertEqual(execution.run_status(run)["state"], "interrupted")
            finally:
                if child.poll() is None:
                    child.communicate(timeout=5)

    def test_missing_worker_registration_has_bounded_grace(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            execution._set_status(run, state="starting", started_at=execution._now())
            self.assertEqual(execution.run_status(run)["state"], "starting")
            old = datetime.now(timezone.utc) - timedelta(seconds=execution.STARTUP_GRACE_SECONDS + 1)
            execution._set_status(run, state="starting", started_at=old.isoformat())
            self.assertEqual(execution.run_status(run)["state"], "interrupted")

    def test_terminal_receipt_is_preserved_after_worker_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            execution._set_status(run, state="completed", started_at=execution._now(), returncode=0)
            self.assertEqual(execution.run_status(run)["state"], "completed")

    def test_worker_prelaunch_failure_records_terminal_status(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            execution._set_status(run, state="starting", started_at=execution._now())
            # Missing approval formerly escaped before the worker's try block.
            self.assertEqual(execution._worker(run), 1)
            status = execution.run_status(run)
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["reason"], "FileNotFoundError")

    def saved_run(self, run):
        session = self.session()
        plan = execution.run_plan(session, 2, 1)
        execution._set_status(run, state="starting", started_at=execution._now())
        execution._write(run / "session.json", session.model_dump(mode="json"))
        execution._write(run / "approval.json", plan)
        execution._write(run / "config.json", plan["config"])
        return plan

    def test_worker_rejects_changed_saved_config_before_solver_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            plan = self.saved_run(run)
            changed = {**plan["config"], "volume_fraction": 0.50}
            execution._write(run / "config.json", changed)
            with patch.object(execution.subprocess, "run") as solve:
                self.assertEqual(execution._worker(run), 1)
                solve.assert_not_called()
            status = execution.run_status(run)
            self.assertEqual(status["state"], "failed")
            self.assertIn("no longer match the approved plan", status["reason"])

    def test_worker_rechecks_saved_session_and_budget_hash(self):
        for changed_file in ("session.json", "approval.json"):
            with self.subTest(changed_file=changed_file), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                self.saved_run(run)
                payload = json.loads((run / changed_file).read_text())
                if changed_file == "session.json":
                    payload["spec"]["geometry"]["parameters"]["hole_radius"]["value"] = 8
                else:
                    payload["max_iterations"] = 3
                execution._write(run / changed_file, payload)
                with patch.object(execution.subprocess, "run") as solve:
                    self.assertEqual(execution._worker(run), 1)
                    solve.assert_not_called()

    def test_worker_accepts_reordered_json_for_unchanged_approved_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            plan = self.saved_run(run)
            reordered = dict(reversed(list(plan["config"].items())))
            execution._write(run / "config.json", reordered)
            with patch.object(execution.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as solve:
                self.assertEqual(execution._worker(run), 0)
                solve.assert_called_once()
            self.assertEqual(execution.run_status(run)["state"], "completed")


if __name__ == "__main__":
    unittest.main()
