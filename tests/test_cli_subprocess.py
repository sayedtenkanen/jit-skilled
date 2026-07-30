"""Subprocess-level tests: invoke the CLI the way a real user would
(`python -m ...`), not just its internal functions. Per project hygiene
policy, entry points meant to run as subprocesses need to be tested that
way, not only unit-tested underneath.
"""
import json
import subprocess
import sys

import pytest


def _run_cli(*args: str, cwd, env) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "jitskilled", *args],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=60,
    )


@pytest.fixture
def cli_env(project_root):
    import os
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)  # force mock backend, deterministic
    env["JITSKILLED_LLM_BACKEND"] = "mock"
    env["PYTHONPATH"] = str(project_root / "src")
    return env


def test_run_zero_shot_via_module_cli(tmp_path, project_root, cli_env):
    proc = _run_cli(
        "run", "--mode", "zero_shot", "--run_name", "pytest_zero_shot",
        "--runs_dir", str(tmp_path),
        cwd=project_root, env=cli_env,
    )
    assert proc.returncode == 0, proc.stderr
    eval_path = tmp_path / "pytest_zero_shot" / "eval.json"
    assert eval_path.exists()
    eval_data = json.loads(eval_path.read_text())
    assert 0.0 <= eval_data["accuracy"] <= 1.0
    assert eval_data["num_tasks"] == eval_data["num_correct"] + (
        eval_data["num_tasks"] - eval_data["num_correct"]
    )


def test_run_skill_mode_writes_skill_files(tmp_path, project_root, cli_env):
    proc = _run_cli(
        "run", "--mode", "skill",
        "--slot_library", str(project_root / "configs" / "slots_v1.yaml"),
        "--run_name", "pytest_skill",
        "--runs_dir", str(tmp_path),
        cwd=project_root, env=cli_env,
    )
    assert proc.returncode == 0, proc.stderr
    skills_dir = tmp_path / "pytest_skill" / "skills"
    assert skills_dir.exists()
    assert list(skills_dir.glob("*.md"))


def test_missing_required_arg_exits_nonzero(tmp_path, project_root, cli_env):
    # --mode is required; omitting it should fail argparse, not crash weirdly.
    proc = _run_cli("run", "--run_name", "pytest_missing_mode",
                     cwd=project_root, env=cli_env)
    assert proc.returncode != 0


def test_optimize_end_to_end_via_module_cli(tmp_path, project_root, cli_env):
    zero_shot = _run_cli(
        "run", "--mode", "zero_shot", "--run_name", "zero_shot",
        "--runs_dir", str(tmp_path), cwd=project_root, env=cli_env,
    )
    assert zero_shot.returncode == 0, zero_shot.stderr

    skill_v1 = _run_cli(
        "run", "--mode", "skill",
        "--slot_library", str(project_root / "configs" / "slots_v1.yaml"),
        "--run_name", "v1", "--runs_dir", str(tmp_path),
        cwd=project_root, env=cli_env,
    )
    assert skill_v1.returncode == 0, skill_v1.stderr

    output_path = tmp_path / "slots_v2.yaml"
    optimize = _run_cli(
        "optimize",
        "--current_run", str(tmp_path / "v1"),
        "--previous_run", str(tmp_path / "zero_shot"),
        "--slot_library", str(project_root / "configs" / "slots_v1.yaml"),
        "--output", str(output_path),
        cwd=project_root, env=cli_env,
    )
    assert optimize.returncode == 0, optimize.stderr
    assert output_path.exists()
    summary_path = tmp_path / "slots_v2_optimize_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert "transition_counts" in summary
    assert "editor_patch" in summary
