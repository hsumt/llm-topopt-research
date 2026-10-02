"""Exercise the launcher with fake Docker and synthetic runtime configuration.

The repository's real runtime configuration and credentials are never opened.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


LAUNCHER_SOURCE = Path(__file__).resolve().parents[1] / "docker" / "compose"


@unittest.skipUnless(LAUNCHER_SOURCE.is_file(), "The host launcher is not packaged in the solver image.")
class DockerLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "project with spaces"
        (self.project / "docker").mkdir(parents=True)
        self.launcher = self.project / "docker" / "compose"
        shutil.copyfile(LAUNCHER_SOURCE, self.launcher)
        self.launcher.chmod(0o755)

        fake_bin = self.root / "fake-bin"
        fake_bin.mkdir()
        self.call_log = self.root / "docker-call.json"
        fake_docker = fake_bin / "docker"
        fake_docker.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys\n"
            "call = {'args': sys.argv[1:], 'cwd': os.getcwd(), "
            "'disable_implicit_env': os.environ.get('COMPOSE_DISABLE_ENV_FILE'), "
            "'host_uid': os.environ.get('HOST_UID'), 'host_gid': os.environ.get('HOST_GID')}\n"
            "pathlib.Path(os.environ['DOCKER_CALL_LOG']).write_text(json.dumps(call))\n",
            encoding="utf-8",
        )
        fake_docker.chmod(0o755)
        # Do not inherit actual service credentials into even this fake process.
        self.environment = {
            "PATH": f"{fake_bin}:{os.defpath}",
            "DOCKER_CALL_LOG": str(self.call_log),
        }

    def run_launcher(self, *args):
        result = subprocess.run(
            [str(self.launcher), *args], cwd=self.root, env=self.environment,
            capture_output=True, text=True, check=True, timeout=10,
        )
        call = json.loads(self.call_log.read_text(encoding="utf-8"))
        self.assertEqual(call["cwd"], str(self.project))
        self.assertEqual(call["disable_implicit_env"], "1")
        self.assertEqual(call["host_uid"], str(os.getuid()))
        self.assertEqual(call["host_gid"], str(os.getgid()))
        self.assertTrue((self.project / "artifacts").is_dir())
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")
        return call

    def test_missing_configuration_explicitly_selects_dev_null(self):
        call = self.run_launcher("up", "--build", "-d")
        self.assertEqual(call["args"], [
            "compose", "--env-file", "/dev/null", "-f", "docker-compose.yml", "up", "--build", "-d",
        ])

    def test_existing_configuration_is_delegated_without_reading_or_sourcing(self):
        marker = self.root / "must-not-be-created"
        # This is synthetic test input, not a credential. It is only written;
        # neither the test nor fake Docker reads it.
        (self.project / ".env").write_text(
            "LAUNCHER_TEST_SENTINEL=synthetic-value-must-not-appear\n"
            f"LAUNCHER_TEST_COMMAND=$(touch '{marker}')\n",
            encoding="utf-8",
        )
        call = self.run_launcher("up", "-d")
        self.assertEqual(call["args"], [
            "compose", "--env-file", ".env", "-f", "docker-compose.yml", "up", "-d",
        ])
        self.assertFalse(marker.exists())
        self.assertNotIn("synthetic-value-must-not-appear", json.dumps(call))

    def test_directory_named_env_is_not_selected_as_a_file(self):
        (self.project / ".env").mkdir()
        call = self.run_launcher("ps")
        self.assertEqual(call["args"][2], "/dev/null")

    def test_forwarded_arguments_remain_literal_and_keep_spaces(self):
        marker = self.root / "argument-must-not-execute"
        args = ["run", "--rm", "app", "python", "-c", "print('hello world')", f"$(touch '{marker}')"]
        call = self.run_launcher(*args)
        self.assertEqual(call["args"][5:], args)
        self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
