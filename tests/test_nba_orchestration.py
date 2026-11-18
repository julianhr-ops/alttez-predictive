import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(os.name == "nt" and shutil.which("powershell"), "Windows PowerShell required")
class NbaOrchestrationTests(unittest.TestCase):
    def test_nba_runs_once_after_soccer_on_both_paths_and_failure_is_nonfatal(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_daily_process.ps1"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            script = Path(shutil.copy2(script, root / "scripts" / script.name))
            log = root / "calls.log"
            (root / "python.cmd").write_text(
                '@echo off\necho %*>>"%NBA_TEST_LOG%"\n'
                'echo %1 | findstr /i "%NBA_FAIL_STAGE%" >nul && exit /b 1\n'
                'echo %1 | findstr /i "smoke_live_pipeline.py" >nul && exit /b %NBA_SMOKE_EXIT%\n'
                'echo %1 | findstr /i "nba_daily.py" >nul && exit /b 1\nexit /b 0\n'
            )
            env = {**os.environ, "PATH": str(root) + os.pathsep + os.environ["PATH"], "NBA_TEST_LOG": str(log),
                   "ALTTEZ_DAILY_BRIEF_DELIVER": "0"}
            for smoke_exit, failure, stage in (
                ("0", "never_a_stage", None),
                ("3", "never_a_stage", None),
                ("0", "generate_daily_brief.py", "committee daily decision brief"),
                ("0", "run_challenger_cycle.py", "challenger cycle"),
            ):
                with self.subTest(smoke_exit=smoke_exit, failure=failure):
                    log.unlink(missing_ok=True)
                    env["NBA_SMOKE_EXIT"] = smoke_exit
                    env["NBA_FAIL_STAGE"] = failure
                    result = subprocess.run(
                        ["powershell", "-NoProfile", "-File", str(script), "-Date", "2026-11-18"],
                        env=env, capture_output=True, text=True, errors="replace", check=False,
                    )
                    calls = log.read_text().splitlines()
                    nba = [index for index, call in enumerate(calls) if "nba_daily.py" in call]
                    self.assertEqual(result.returncode == 0, stage is None, result.stdout + result.stderr)
                    self.assertEqual(len(nba), 1)
                    self.assertTrue(all(str(root) in call for call in calls))
                    self.assertIn("shadow_readiness_check.py", calls[0])
                    if stage:
                        alert = next(index for index, call in enumerate(calls)
                                     if "send_pipeline_alert.py" in call and stage in call and "critical" in call)
                        self.assertLess(alert, nba[0])
                    elif smoke_exit == "0":
                        challenger = next(index for index, call in enumerate(calls) if "run_challenger_cycle.py" in call)
                        self.assertGreater(nba[0], challenger)
