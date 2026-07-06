#!/usr/bin/env python3
"""Analyze archived MLE-STAR experiment runs and write aggregated reports.

For executing experiment runs, use run_experiments.py instead to seperate concerns (recommended);
But it's also possible to run only this script. 

Examples:
  python automated_evaluation/run_experiments.py --dry-run --tasks spaceship-titanic
  python automated_evaluation/evaluate.py --summarize-only --runs-root automated_evaluation/runs/<stamp>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import statistics
import sys
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "eval_results"
RUNS_DIR = EVAL_DIR / "runs"
MANIFEST_PATH = EVAL_DIR / "tasks_manifest.json"
TASKS_DIR = (
    REPO_ROOT
    / "agents"
    / "machine-learning-engineering"
    / "machine_learning_engineering"
    / "tasks"
)
IMPROVED_AGENT_DIR = REPO_ROOT / "agents" / "machine-learning-engineering"
DEFAULT_VANILLA_AGENT_DIR = REPO_ROOT.parent / "mle-star_vanilla" / "agents" / "machine-learning-engineering"
CONFIG_REL = Path("machine_learning_engineering/shared_libraries/config.py")
TEST_SCRIPTS = REPO_ROOT / "test-scripts"
USER_PROMPT = "execute your given task"

sys.path.insert(0, str(TEST_SCRIPTS))
from analyze_run import analyze_run, to_row  # noqa: E402
from aggregate_runs import find_run_dirs, labels_for  # noqa: E402

def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def task_is_complete(task_name: str) -> bool:
    folder = TASKS_DIR / task_name
    return (
        (folder / "train.csv").is_file()
        and (folder / "test.csv").is_file()
        and (folder / "task_description.txt").is_file()
    )


SYSTEM_SPECS: dict[str, dict[str, Any]] = {
    "improved": {
        "archive_label": "skrub-full",
        "table_report_enabled": True,
        "tuning_enabled": True,
    },
    "vanilla": {
        "archive_label": "vanilla",
        "table_report_enabled": False,
        "tuning_enabled": False,
    },
}


@dataclass
class TaskSpec:
    task_name: str
    task_type: str
    lower: bool
    metric: str = ""
    ready: bool = True


@dataclass
class SystemSpec:
    key: str
    archive_label: str
    agent_dir: Path
    table_report_enabled: bool
    tuning_enabled: bool


@dataclass
class EvalConfig:
    tasks: list[str]
    systems: list[str]
    repeats: list[str]
    seed: int
    model_label: str
    improved_agent_dir: Path
    vanilla_agent_dir: Path
    runs_root: Path
    results_dir: Path
    sync_tasks_to_vanilla: bool
    uv_sync_before_run: bool
    dry_run: bool
    skip_existing: bool
    summarize_only: bool
    verbose: bool = False
    live_log: bool = False


def _format_wall(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes}m"


def _emit_status(config: EvalConfig, message: str, *, err: bool = False) -> None:
    if not config.verbose:
        return
    print(message, file=sys.stderr if err else sys.stdout, flush=True)


def _count_matrix_cells(
    tasks: list[TaskSpec],
    systems: list[SystemSpec],
    repeats: list[str],
) -> int:
    total = 0
    for task in tasks:
        if not task.ready:
            total += 1
        else:
            total += len(systems) * len(repeats)
    return total


def print_experiment_banner(
    config: EvalConfig,
    tasks: list[TaskSpec],
    *,
    total_cells: int,
) -> None:
    if not config.verbose:
        return
    task_names = [t.task_name for t in tasks]
    lines = [
        "",
        "=" * 72,
        "MLE-STAR experiment batch",
        "=" * 72,
        f"Runs root:   {config.runs_root}",
        f"Tasks:       {len(tasks)} ({', '.join(task_names) if len(task_names) <= 4 else task_names[0] + ', ...'})",
        f"Systems:     {', '.join(config.systems)}",
        f"Repeats:     {', '.join(config.repeats)}",
        f"Matrix size: {total_cells} cell(s)",
        f"Seed:        {config.seed}",
        f"Model:       {config.model_label}",
        f"Live ADK log: {'yes' if config.live_log else 'no'}",
        f"Dry run:     {config.dry_run}",
    ]
    if config.skip_existing:
        lines.append("Skip existing archives: yes")
    if config.uv_sync_before_run:
        lines.append("uv sync before each run: yes")
    lines.append("=" * 72)
    _emit_status(config, "\n".join(lines))


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _git_sha(cwd: Path | None = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd or REPO_ROOT,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _model_slug(model_label: str) -> str:
    return model_label.split("/", 1)[-1]


def _load_dotenv(agent_dir: Path) -> dict[str, str]:
    env_path = agent_dir / ".env"
    if not env_path.is_file():
        return {}
    env: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if value and value[0] in "'\"":
            quote = value[0]
            end = value.find(quote, 1)
            if end != -1:
                value = value[1:end]
            else:
                value = value.strip("'\"")
        else:
            value = value.split("#", 1)[0].strip().strip("'\"")
        if value:
            env[key.strip()] = value
    return env


def _manifest_lookup() -> dict[str, dict[str, Any]]:
    if not MANIFEST_PATH.is_file():
        return {}
    manifest = load_manifest(MANIFEST_PATH)
    return {spec["task_name"]: spec for spec in manifest.get("tasks", [])}


def scan_tasks(tasks_dir: Path = TASKS_DIR) -> list[TaskSpec]:
    """Discover runnable tasks from the agent tasks directory."""
    if not tasks_dir.is_dir():
        raise FileNotFoundError(f"Tasks directory not found: {tasks_dir}")

    manifest_by_name = _manifest_lookup()
    discovered: list[TaskSpec] = []
    for entry in sorted(tasks_dir.iterdir()):
        if not entry.is_dir():
            continue
        task_name = entry.name
        ready = task_is_complete(task_name)
        task_config_path = entry / "task_config.json"
        if task_config_path.is_file():
            cfg = json.loads(task_config_path.read_text(encoding="utf-8"))
            discovered.append(
                TaskSpec(
                    task_name=task_name,
                    task_type=cfg.get("task_type", "Tabular Classification"),
                    lower=bool(cfg.get("lower", False)),
                    metric=cfg.get("metric", ""),
                    ready=ready,
                )
            )
            continue
        manifest_spec = manifest_by_name.get(task_name)
        if manifest_spec:
            discovered.append(
                TaskSpec(
                    task_name=task_name,
                    task_type=manifest_spec["task_type"],
                    lower=bool(manifest_spec["lower"]),
                    metric=manifest_spec.get("metric", ""),
                    ready=ready,
                )
            )
            continue
        discovered.append(
            TaskSpec(
                task_name=task_name,
                task_type="Tabular Classification",
                lower=False,
                ready=ready,
            )
        )
    return discovered


def _replace_config_field(
    text: str,
    field: str,
    value: str | int | bool,
    *,
    required: bool = True,
) -> str:
    if isinstance(value, bool):
        patterns = [(rf"({field}: bool = )(True|False)", rf"\1{'True' if value else 'False'}")]
    elif isinstance(value, int):
        patterns = [(rf"({field}: int = )\d+", rf"\1{value}")]
    else:
        value_str = str(value)
        patterns = [
            (
                rf"({field}: str = )\([^)]+\)",
                rf'\1(\n        "{value_str}"\n    )',
            ),
            (
                rf'({field}: str = )"[^"]*"',
                rf'\1"{value_str}"',
            ),
        ]
    for pattern, repl in patterns:
        if isinstance(value, int):
            repl = lambda m, v=value: f"{m.group(1)}{v}"
            new_text, count = re.subn(pattern, repl, text, count=1, flags=re.DOTALL)
        else:
            new_text, count = re.subn(pattern, repl, text, count=1, flags=re.DOTALL)
        if count:
            return new_text
    if not required:
        return text
    raise ValueError(f"Could not patch config field: {field}")


@contextmanager
def patched_agent_config(
    agent_dir: Path,
    *,
    task_name: str,
    task_type: str,
    lower: bool,
    seed: int,
    table_report_enabled: bool,
    tuning_enabled: bool,
) -> Iterator[None]:
    config_path = agent_dir / CONFIG_REL
    if not config_path.is_file():
        raise FileNotFoundError(f"Agent config not found: {config_path}")
    original = config_path.read_text(encoding="utf-8")
    patched = original
    patched = _replace_config_field(patched, "task_name", task_name)
    patched = _replace_config_field(patched, "task_type", task_type)
    patched = _replace_config_field(patched, "lower", lower)
    patched = _replace_config_field(patched, "seed", seed)
    # Vanilla baseline config omits skrub-only flags; skip when absent.
    patched = _replace_config_field(
        patched, "table_report_enabled", table_report_enabled, required=False
    )
    patched = _replace_config_field(
        patched, "tuning_enabled", tuning_enabled, required=False
    )
    patched = _replace_config_field(
        patched, "use_data_leakage_checker", True, required=False
    )
    config_path.write_text(patched, encoding="utf-8")
    try:
        yield
    finally:
        config_path.write_text(original, encoding="utf-8")


def sync_tasks_to_agent(agent_dir: Path, tasks: list[str] | None = None) -> None:
    src_root = TASKS_DIR
    dst_root = agent_dir / "machine_learning_engineering" / "tasks"
    dst_root.mkdir(parents=True, exist_ok=True)
    names = tasks or [p.name for p in scan_tasks()]
    for task_name in names:
        src = src_root / task_name
        dst = dst_root / task_name
        if not src.is_dir():
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def _iso_duration_seconds(started_at: str | None, finished_at: str | None) -> float | None:
    if not started_at or not finished_at:
        return None
    try:
        start = datetime.fromisoformat(started_at)
        end = datetime.fromisoformat(finished_at)
        return max(0.0, (end - start).total_seconds())
    except (TypeError, ValueError):
        return None


def _read_meta_wall_seconds(run_dir: Path) -> float | None:
    meta_path = run_dir / "meta.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    wall = meta.get("wall_seconds")
    if wall is not None:
        return float(wall)
    return _iso_duration_seconds(meta.get("started_at"), meta.get("finished_at"))


def _workspace_dir(agent_dir: Path, task_name: str) -> Path:
    return agent_dir / "machine_learning_engineering" / "workspace" / task_name


def _archive_dir(
    runs_root: Path,
    *,
    task_name: str,
    system_label: str,
    model_label: str,
    repeat: str,
) -> Path:
    return runs_root / task_name / system_label / _model_slug(model_label) / repeat


def _run_bundle_complete(archive_path: Path) -> bool:
    return (archive_path / "final_state.json").is_file()


_PROMPT_MARKER = "[user]:"
_ABORTED_LINE = "Aborted!"


def _archived_run_log(archive_path: Path) -> Path | None:
    logs = sorted(archive_path.glob("adk_run_*.log"))
    return logs[-1] if logs else None


def _log_returned_to_prompt(log_path: Path) -> bool:
    """True when ADK finished and reprinted the interactive prompt."""
    if not log_path.is_file():
        return False
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if text.count(_PROMPT_MARKER) < 2:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    for line in reversed(lines[-6:]):
        if line == _ABORTED_LINE:
            continue
        return line.startswith(_PROMPT_MARKER)
    return False


def _run_succeeded(archive_path: Path, log_path: Path | None = None) -> bool:
    if not _run_bundle_complete(archive_path):
        return False
    log = log_path if log_path is not None else _archived_run_log(archive_path)
    if log is None:
        return False
    return _log_returned_to_prompt(log)


def _run_failure_reason(
    *,
    has_final_state: bool,
    returned_to_prompt: bool,
) -> str | None:
    reasons: list[str] = []
    if not has_final_state:
        reasons.append("missing final_state.json")
    if not returned_to_prompt:
        reasons.append("ADK did not return to [user]: prompt")
    return "; ".join(reasons) if reasons else None


def summarize_run_outcomes(run_log: list[dict[str, Any]]) -> dict[str, int]:
    ok = sum(
        1
        for row in run_log
        if row.get("status") in {"completed", "completed_with_errors"}
    )
    failed = sum(1 for row in run_log if row.get("status") == "failed")
    skipped = sum(1 for row in run_log if row.get("status") == "skipped")
    planned = sum(1 for row in run_log if row.get("status") == "planned")
    return {
        "ok": ok,
        "failed": failed,
        "skipped": skipped,
        "planned": planned,
    }


def _maybe_uv_sync(agent_dir: Path) -> None:
    subprocess.run(
        ["uv", "sync"],
        cwd=agent_dir,
        check=True,
        capture_output=True,
        text=True,
    )


def run_agent_once(
    agent_dir: Path,
    *,
    task: TaskSpec,
    system: SystemSpec,
    seed: int,
    log_path: Path,
    uv_sync_before_run: bool = False,
    live_log: bool = False,
) -> subprocess.CompletedProcess[str]:
    agent_dir = agent_dir.resolve()
    env = os.environ.copy()
    env.update(_load_dotenv(agent_dir))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["uv", "run", "adk", "run", "machine_learning_engineering"]

    if uv_sync_before_run:
        _maybe_uv_sync(agent_dir)

    with patched_agent_config(
        agent_dir,
        task_name=task.task_name,
        task_type=task.task_type,
        lower=task.lower,
        seed=seed,
        table_report_enabled=system.table_report_enabled,
        tuning_enabled=system.tuning_enabled,
    ):
        workspace = _workspace_dir(agent_dir, task.task_name)
        if workspace.exists():
            shutil.rmtree(workspace)

        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write(
                f"# run_experiments.py agent run\n"
                f"# task={task.task_name} system={system.key} seed={seed}\n"
                f"# cwd={agent_dir}\n"
                f"# cmd={' '.join(cmd)}\n\n"
            )
            log_handle.flush()

            proc = subprocess.Popen(
                cmd,
                cwd=agent_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            assert proc.stdin is not None
            proc.stdin.write(f"{USER_PROMPT}\n")
            proc.stdin.close()

            assert proc.stdout is not None
            for line in proc.stdout:
                log_handle.write(line)
                log_handle.flush()
                if live_log:
                    sys.stdout.write(line)
                    sys.stdout.flush()

            returncode = proc.wait()

    return subprocess.CompletedProcess(args=cmd, returncode=returncode)


def archive_run_bundle(
    agent_dir: Path,
    *,
    task_name: str,
    archive_path: Path,
    meta: dict[str, Any],
    log_path: Path | None,
) -> None:
    workspace = _workspace_dir(agent_dir, task_name)
    if archive_path.exists():
        shutil.rmtree(archive_path)
    archive_path.mkdir(parents=True, exist_ok=True)

    final_state = workspace / "final_state.json"
    if final_state.is_file():
        shutil.copy2(final_state, archive_path / "final_state.json")

    table_report = workspace / "table_report.json"
    if table_report.is_file():
        shutil.copy2(table_report, archive_path / "table_report.json")

    for sub in ("1", "ensemble"):
        src = workspace / sub
        if src.is_dir():
            shutil.copytree(src, archive_path / sub)

    for pattern in ("adk_run_*.log",):
        for src in sorted(agent_dir.glob(pattern)):
            shutil.copy2(src, archive_path / src.name)
        for src in sorted((agent_dir / "run-logs").glob(pattern)):
            shutil.copy2(src, archive_path / src.name)
        for src in sorted(workspace.glob(pattern)):
            shutil.copy2(src, archive_path / src.name)

    if log_path and log_path.is_file():
        shutil.copy2(log_path, archive_path / log_path.name)

    (archive_path / "meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    analysis = analyze_run(archive_path)
    if not analysis.extras.get("error"):
        (archive_path / "analysis.json").write_text(
            json.dumps(
                {
                    "row": to_row(analysis),
                    "scores": analysis.scores,
                    "extras": analysis.extras,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def build_system_specs(config: EvalConfig) -> list[SystemSpec]:
    specs: list[SystemSpec] = []
    for key in config.systems:
        if key not in SYSTEM_SPECS:
            raise ValueError(f"Unknown system: {key}. Choose from: {sorted(SYSTEM_SPECS)}")
        base = SYSTEM_SPECS[key]
        agent_dir = (
            config.improved_agent_dir if key == "improved" else config.vanilla_agent_dir
        )
        specs.append(
            SystemSpec(
                key=key,
                archive_label=base["archive_label"],
                agent_dir=agent_dir.resolve(),
                table_report_enabled=base["table_report_enabled"],
                tuning_enabled=base["tuning_enabled"],
            )
        )
    return specs


def validate_runtime(config: EvalConfig, systems: list[SystemSpec]) -> None:
    if not config.improved_agent_dir.is_dir():
        raise FileNotFoundError(
            f"Improved agent directory not found: {config.improved_agent_dir}"
        )
    for system in systems:
        if not system.agent_dir.is_dir():
            raise FileNotFoundError(
                f"Agent directory for system '{system.key}' not found: {system.agent_dir}. "
                "Bootstrap vanilla with vanilla-scripts/bootstrap_vanilla_baseline.sh "
                "(sklearn-only prompts) or revert_vanilla_prompts.sh on an existing worktree; "
                "or pass --vanilla-agent-dir."
            )


def execute_evaluation_matrix(config: EvalConfig) -> list[dict[str, Any]]:
    tasks = scan_tasks()
    if config.tasks:
        wanted = set(config.tasks)
        tasks = [task for task in tasks if task.task_name in wanted]
    if not tasks:
        raise SystemExit("No tasks matched the selection.")

    systems = build_system_specs(config)
    if not config.dry_run:
        validate_runtime(config, systems)

    if config.sync_tasks_to_vanilla:
        sync_tasks_to_agent(config.improved_agent_dir, [t.task_name for t in tasks])
        sync_tasks_to_agent(config.vanilla_agent_dir, [t.task_name for t in tasks])

    total_cells = _count_matrix_cells(tasks, systems, config.repeats)
    print_experiment_banner(config, tasks, total_cells=total_cells)

    run_log: list[dict[str, Any]] = []
    cell_index = 0
    progress_bar = None
    if config.verbose and not config.live_log and not config.dry_run:
        try:
            from tqdm import tqdm

            progress_bar = tqdm(
                total=total_cells,
                desc="Experiment runs",
                unit="cell",
            )
        except ImportError:
            progress_bar = None

    for task in tasks:
        if not task.ready:
            cell_index += 1
            if progress_bar is not None:
                progress_bar.update(1)
            print(
                f"Skipping incomplete task pack: {task.task_name}",
                file=sys.stderr,
            )
            _emit_status(
                config,
                f"[{cell_index}/{total_cells}] SKIP   {task.task_name} (incomplete task pack)",
            )
            run_log.append(
                {
                    "task_name": task.task_name,
                    "status": "skipped",
                    "reason": "incomplete_task_pack",
                }
            )
            continue

        for system in systems:
            if system.key == "vanilla" and config.sync_tasks_to_vanilla:
                sync_tasks_to_agent(config.vanilla_agent_dir, [task.task_name])

            for repeat in config.repeats:
                cell_index += 1
                remaining = total_cells - cell_index
                archive_path = _archive_dir(
                    config.runs_root,
                    task_name=task.task_name,
                    system_label=system.archive_label,
                    model_label=config.model_label,
                    repeat=repeat,
                )
                entry = {
                    "task_name": task.task_name,
                    "system": system.key,
                    "archive_label": system.archive_label,
                    "repeat": repeat,
                    "archive_path": str(archive_path.relative_to(REPO_ROOT)),
                }
                if config.skip_existing and _run_succeeded(archive_path):
                    entry["status"] = "skipped"
                    entry["reason"] = "existing_archive"
                    run_log.append(entry)
                    msg = (
                        f"[{cell_index}/{total_cells}] SKIP   "
                        f"{task.task_name} / {system.key} / {repeat} "
                        f"(existing archive, {remaining} remaining)"
                    )
                    _emit_status(config, msg)
                    if not config.verbose:
                        print(f"Skip existing: {archive_path}")
                    if progress_bar is not None:
                        progress_bar.update(1)
                    continue

                if config.dry_run:
                    entry["status"] = "planned"
                    run_log.append(entry)
                    msg = (
                        f"[{cell_index}/{total_cells}] PLAN   "
                        f"{task.task_name} / {system.key} / {repeat} "
                        f"({remaining} remaining)"
                    )
                    _emit_status(config, msg)
                    if not config.verbose:
                        print(f"Would run: {task.task_name} / {system.key} / {repeat}")
                    continue

                started_at = datetime.now(timezone.utc).isoformat()
                with tempfile.TemporaryDirectory(prefix="mle-eval-") as tmp:
                    log_path = Path(tmp) / f"adk_run_{_utc_stamp()}.log"
                    _emit_status(
                        config,
                        "\n"
                        + "-" * 72
                        + f"\n[{cell_index}/{total_cells}] START  "
                        f"{task.task_name} / {system.key} / {repeat}  "
                        f"({remaining} remaining after this)\n"
                        f"  archive -> {archive_path.relative_to(REPO_ROOT)}\n"
                        f"  agent   -> {system.agent_dir}\n"
                        f"  cmd     -> uv run adk run machine_learning_engineering\n"
                        + "-" * 72,
                    )
                    if not config.verbose:
                        print(
                            f"Running {task.task_name} / {system.key} / {repeat} "
                            f"-> {archive_path}"
                        )
                    try:
                        proc = run_agent_once(
                            system.agent_dir,
                            task=task,
                            system=system,
                            seed=config.seed,
                            log_path=log_path,
                            uv_sync_before_run=config.uv_sync_before_run,
                            live_log=config.live_log,
                        )
                        finished_at = datetime.now(timezone.utc).isoformat()
                        wall_seconds = _iso_duration_seconds(started_at, finished_at)
                        meta = {
                            "task": task.task_name,
                            "task_type": task.task_type,
                            "metric": task.metric,
                            "lower_is_better": task.lower,
                            "system": system.archive_label,
                            "system_key": system.key,
                            "repeat": repeat,
                            "git_sha": _git_sha(system.agent_dir),
                            "repo": system.agent_dir.parents[1].name,
                            "agent_dir": str(system.agent_dir),
                            "model": config.model_label,
                            "seed": config.seed,
                            "table_report_enabled": system.table_report_enabled,
                            "tuning_enabled": system.tuning_enabled,
                            "use_data_leakage_checker": True,
                            "uv_sync_before_run": config.uv_sync_before_run,
                            "started_at": started_at,
                            "finished_at": finished_at,
                            "wall_seconds": (
                                round(wall_seconds, 1) if wall_seconds is not None else None
                            ),
                            "exit_code": proc.returncode,
                            "log_file": log_path.name,
                            "user_prompt": USER_PROMPT,
                        }
                        archive_run_bundle(
                            system.agent_dir,
                            task_name=task.task_name,
                            archive_path=archive_path,
                            meta=meta,
                            log_path=log_path,
                        )
                        archived_log = archive_path / log_path.name
                        entry["has_final_state"] = _run_bundle_complete(archive_path)
                        entry["returned_to_prompt"] = _log_returned_to_prompt(archived_log)
                        if _run_succeeded(archive_path, archived_log):
                            entry["status"] = (
                                "completed"
                                if proc.returncode == 0
                                else "completed_with_errors"
                            )
                        else:
                            entry["status"] = "failed"
                            entry["error"] = _run_failure_reason(
                                has_final_state=entry["has_final_state"],
                                returned_to_prompt=entry["returned_to_prompt"],
                            )
                        entry["exit_code"] = proc.returncode
                        if entry["status"] == "failed":
                            status_word = "FAIL"
                        elif proc.returncode == 0:
                            status_word = "DONE"
                        else:
                            status_word = "DONE (errors)"
                        completion_note = (
                            "final_state=yes  prompt=yes"
                            if entry["has_final_state"] and entry["returned_to_prompt"]
                            else (
                                f"final_state={'yes' if entry['has_final_state'] else 'no'}  "
                                f"prompt={'yes' if entry['returned_to_prompt'] else 'no'}"
                            )
                        )
                        _emit_status(
                            config,
                            f"[{cell_index}/{total_cells}] {status_word}  "
                            f"{task.task_name} / {system.key} / {repeat}  "
                            f"exit={proc.returncode}  wall={_format_wall(wall_seconds)}  "
                            f"{completion_note}  ({remaining} remaining)",
                            err=entry["status"] == "failed",
                        )
                        if entry["status"] == "failed" and not config.verbose:
                            print(
                                f"Run incomplete: {entry.get('error', 'unknown')} "
                                f"({archive_path})",
                                file=sys.stderr,
                            )
                    except Exception as exc:  # noqa: BLE001 - per-run failure should not abort matrix
                        entry["status"] = "failed"
                        entry["error"] = str(exc)
                        _emit_status(
                            config,
                            f"[{cell_index}/{total_cells}] FAIL   "
                            f"{task.task_name} / {system.key} / {repeat}  "
                            f"error={exc}  ({remaining} remaining)",
                            err=True,
                        )
                        if not config.verbose:
                            print(f"Run failed: {exc}", file=sys.stderr)
                run_log.append(entry)
                if progress_bar is not None:
                    progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.close()

    if run_log and not config.dry_run:
        config.runs_root.mkdir(parents=True, exist_ok=True)
        (config.runs_root / "run_matrix.json").write_text(
            json.dumps(run_log, indent=2),
            encoding="utf-8",
        )
    return run_log


def discover_run_dirs(runs_root: Path) -> list[Path]:
    if not runs_root.is_dir():
        return []
    return find_run_dirs(runs_root)


def analyze_runs(run_dirs: list[Path], runs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        analysis = analyze_run(run_dir)
        if analysis.extras.get("error"):
            continue
        labels = labels_for(run_dir, runs_root)
        row = {
            **labels,
            "archive_path": str(run_dir.relative_to(REPO_ROOT)),
            "run_path": str(run_dir),
            **to_row(analysis),
        }
        if row.get("wall_seconds") is None:
            wall = _read_meta_wall_seconds(run_dir)
            if wall is not None:
                row["wall_seconds"] = round(wall, 1)
        rows.append(row)
    return rows


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], None
    return statistics.mean(values), statistics.stdev(values)


def summarize_by_task_system(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        variant = row.get("variant", "")
        system = variant.split("/")[0] if variant else "unknown"
        groups[(row["task"], system)].append(row)

    summary: list[dict[str, Any]] = []
    for (task, system), group in sorted(groups.items()):
        scores = [
            float(r["score_submission"])
            for r in group
            if r.get("score_submission") is not None
        ]
        walls = [
            float(r["wall_seconds"])
            for r in group
            if r.get("wall_seconds") is not None
        ]
        execs = [
            float(r["exec_seconds"])
            for r in group
            if r.get("exec_seconds") is not None
        ]
        dataops = [
            float(r["dataops_adherence"])
            for r in group
            if r.get("dataops_adherence") is not None
        ]
        mean_score, std_score = _mean_std(scores)
        mean_wall, std_wall = _mean_std(walls)
        mean_exec, std_exec = _mean_std(execs)
        mean_dataops, std_dataops = _mean_std(dataops)
        lower = group[0].get("lower_is_better")
        summary.append(
            {
                "task": task,
                "system": system,
                "n_runs": len(group),
                "lower_is_better": lower,
                "score_submission_mean": mean_score,
                "score_submission_std": std_score,
                "wall_seconds_mean": mean_wall,
                "wall_seconds_std": std_wall,
                "exec_seconds_mean": mean_exec,
                "exec_seconds_std": std_exec,
                "dataops_adherence_mean": mean_dataops,
                "dataops_adherence_std": std_dataops,
            }
        )
    return summary


def compare_systems(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summary_rows:
        by_task[row["task"]][row["system"]] = row

    comparisons: list[dict[str, Any]] = []
    for task, systems in sorted(by_task.items()):
        vanilla = systems.get("vanilla")
        skrub = systems.get("skrub-full")
        if not vanilla or not skrub:
            continue
        v_score = vanilla.get("score_submission_mean")
        s_score = skrub.get("score_submission_mean")
        lower = vanilla.get("lower_is_better")
        if v_score is None or s_score is None:
            delta = None
            winner = None
        elif lower:
            delta = v_score - s_score
            winner = "skrub-full" if s_score < v_score else "vanilla"
        else:
            delta = s_score - v_score
            winner = "skrub-full" if s_score > v_score else "vanilla"
        comparisons.append(
            {
                "task": task,
                "vanilla_mean": v_score,
                "skrub_full_mean": s_score,
                "delta_skrub_minus_vanilla_signed": delta,
                "winner": winner,
                "lower_is_better": lower,
            }
        )
    return comparisons


def _fmt_cell(value: Any, *, decimals: int = 4) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if abs(value) >= 1e8:
            return f"{value:.3e}"
        formatted = f"{value:.{decimals}f}"
        if "." in formatted:
            formatted = formatted.rstrip("0").rstrip(".")
        return formatted
    text = str(value).replace("|", "\\|")
    if text.startswith("automated_evaluation/"):
        return text.removeprefix("automated_evaluation/")
    if "/automated_evaluation/" in text:
        return text.split("/automated_evaluation/", 1)[-1]
    return text


def _markdown_table(
    rows: list[dict[str, Any]],
    columns: list[tuple[str, str, str]],
    *,
    decimals: int = 4,
) -> list[str]:
    if not rows:
        return ["_No data._"]
    lines = [
        "| " + " | ".join(label for _, label, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = [
            _fmt_cell(row.get(key), decimals=decimals) for key, _, _ in columns
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _column_legend(columns: list[tuple[str, str, str]]) -> list[str]:
    return [
        "",
        "**Column guide:**",
        *[f"- **{label}** — {description}" for _, label, description in columns],
    ]


SCANNED_TASKS_COLUMNS: list[tuple[str, str, str]] = [
    ("task_name", "Task", "Task folder name under `tasks/`."),
    ("task_type", "Type", "Problem type (regression or classification)."),
    ("metric", "Metric", "Primary metric named in the task pack (e.g. RMSE, accuracy)."),
    ("complete", "Ready", "`yes` if `train.csv`, `test.csv`, and `task_description.txt` exist."),
]

RUN_MATRIX_COLUMNS: list[tuple[str, str, str]] = [
    ("task_name", "Task", "Task executed in this matrix cell."),
    ("system", "System", "Agent variant: `vanilla` (baseline) or `skrub-full` (improved)."),
    ("repeat", "Repeat", "Repeat label for this run (e.g. `run1`)."),
    ("status", "Status", "Orchestrator outcome (`completed`, `completed_with_errors`, `failed`, `skipped`, …). A successful run requires `final_state.json` and ADK returning to `[user]:` in the log."),
    ("archive_path", "Archive", "Relative path to the archived run bundle under `runs/`."),
]

COMPARISON_TABLE_COLUMNS: list[tuple[str, str, str]] = [
    ("task", "Task", "Benchmark task name."),
    (
        "vanilla_mean",
        "Vanilla",
        "Mean submission holdout score across vanilla repeats.",
    ),
    (
        "skrub_full_mean",
        "Skrub-full",
        "Mean submission holdout score across skrub-full repeats.",
    ),
    (
        "delta_skrub_minus_vanilla_signed",
        "Delta",
        "Signed skrub advantage: for **lower-is-better** tasks, `vanilla − skrub` (positive ⇒ vanilla better); for **higher-is-better**, `skrub − vanilla` (positive ⇒ skrub better).",
    ),
    (
        "winner",
        "Winner",
        "System with the better mean submission score on this task.",
    ),
]

SUMMARY_TABLE_COLUMNS: list[tuple[str, str, str]] = [
    ("task", "Task", "Benchmark task name."),
    ("system", "System", "Agent variant aggregated (`vanilla` or `skrub-full`)."),
    ("n_runs", "N", "Number of archived runs in this (task, system) group."),
    (
        "score_submission_mean",
        "Score mean",
        "Mean **submission** holdout score — the primary end-to-end metric.",
    ),
    (
        "score_submission_std",
        "Score std",
        "Standard deviation of submission scores across repeats (`-` when *N* = 1).",
    ),
    (
        "wall_seconds_mean",
        "Wall mean (s)",
        "Mean end-to-end wall-clock time per run (`started_at` → `finished_at` in `meta.json`; includes LLM latency).",
    ),
    (
        "wall_seconds_std",
        "Wall std",
        "Standard deviation of wall-clock times across repeats (`-` when *N* = 1).",
    ),
    (
        "exec_seconds_mean",
        "Exec mean (s)",
        "Mean total Python script execution time per run (excludes LLM latency).",
    ),
    (
        "dataops_adherence_mean",
        "DataOps mean",
        "Mean DataOps adherence in `[0, 1]`: share of skrub DataOps patterns in workspace scripts.",
    ),
]

ALL_RUNS_TABLE_COLUMNS: list[tuple[str, str, str]] = [
    ("task", "Task", "Benchmark task name."),
    ("system", "System", "Agent variant for this run (`vanilla` or `skrub-full`)."),
    ("run", "Run", "Repeat label (e.g. `run1`)."),
    (
        "score_submission",
        "Submission",
        "Holdout score from the **submission** stage — primary comparison metric.",
    ),
    (
        "score_init",
        "Init",
        "Best holdout score after the **initialization** stage.",
    ),
    (
        "score_refine_promoted",
        "Refine",
        "Holdout score of the solution **promoted after refinement**.",
    ),
    (
        "score_tune_search",
        "Tune",
        "Best holdout score from **tuning search** (`-` for vanilla).",
    ),
    (
        "gain_total",
        "Gain",
        "Signed improvement from init to final best (positive = better).",
    ),
    (
        "dataops_adherence",
        "DataOps",
        "DataOps adherence `[0, 1]` for this run's workspace scripts.",
    ),
    (
        "exec_seconds",
        "Exec (s)",
        "Total seconds executing generated Python scripts (not full wall-clock time).",
    ),
    (
        "wall_seconds",
        "Wall (s)",
        "End-to-end wall-clock seconds for this run (orchestrator `started_at` → `finished_at`).",
    ),
    (
        "debug_refinement",
        "Dbg refine",
        "Refinement-stage debug agent events in the ADK log.",
    ),
    (
        "debug_tuning",
        "Dbg tune",
        "Tuning-stage debug agent events in the ADK log.",
    ),
]


def _all_runs_display_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    display: list[dict[str, Any]] = []
    for row in run_rows:
        variant = row.get("variant", "")
        system = variant.split("/")[0] if variant else "-"
        display.append({**row, "system": system})
    return display


def task_status_payload(tasks: list[TaskSpec]) -> dict[str, Any]:
    rows = [
        {
            "task_name": task.task_name,
            "complete": task.ready,
            "task_type": task.task_type,
            "metric": task.metric,
            "lower": task.lower,
        }
        for task in tasks
    ]
    return {
        "task_count": len(rows),
        "complete_count": sum(1 for row in rows if row["complete"]),
        "tasks": rows,
    }


def write_markdown_report(
    path: Path,
    *,
    config: EvalConfig,
    task_status: dict[str, Any],
    run_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    run_log: list[dict[str, Any]] | None,
) -> None:
    lines = [
        "# MLE-STAR automated evaluation report",
        "",
        f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Git SHA: `{_git_sha()}`",
        f"- Runs root: `{config.runs_root.relative_to(REPO_ROOT)}`",
        f"- Seed: `{config.seed}`",
        f"- Model: `{config.model_label}`",
        "",
        "## Scanned tasks",
        "",
        f"Complete: **{task_status['complete_count']} / {task_status['task_count']}**",
        "",
        "| " + " | ".join(label for _, label, _ in SCANNED_TASKS_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in SCANNED_TASKS_COLUMNS) + " |",
    ]
    for task in task_status["tasks"]:
        ready = "yes" if task["complete"] else "no"
        metric = task["metric"] or "-"
        row = {
            "task_name": task["task_name"],
            "task_type": task["task_type"],
            "metric": metric,
            "complete": ready,
        }
        cells = [_fmt_cell(row.get(key)) for key, _, _ in SCANNED_TASKS_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(_column_legend(SCANNED_TASKS_COLUMNS))

    if run_log:
        outcomes = summarize_run_outcomes(run_log)
        lines.extend(
            [
                "",
                "## Run matrix",
                "",
                f"OK: **{outcomes['ok']}** · Failed: **{outcomes['failed']}** · Skipped: **{outcomes['skipped']}** · Planned: **{outcomes['planned']}**",
                "",
                "| " + " | ".join(label for _, label, _ in RUN_MATRIX_COLUMNS) + " |",
                "| " + " | ".join("---" for _ in RUN_MATRIX_COLUMNS) + " |",
            ]
        )
        for row in run_log:
            archive = row.get("archive_path", "")
            if isinstance(archive, str) and archive.startswith("automated_evaluation/"):
                archive = archive.removeprefix("automated_evaluation/")
            matrix_row = {
                "task_name": row.get("task_name", "-"),
                "system": row.get("archive_label", row.get("system", "-")),
                "repeat": row.get("repeat", "-"),
                "status": row.get("status", "-"),
                "archive_path": archive or "-",
            }
            cells = [
                _fmt_cell(matrix_row.get(key)) for key, _, _ in RUN_MATRIX_COLUMNS
            ]
            if matrix_row["archive_path"] not in ("-", ""):
                cells[-1] = f"`{matrix_row['archive_path']}`"
            lines.append("| " + " | ".join(cells) + " |")
        lines.extend(_column_legend(RUN_MATRIX_COLUMNS))

    lines.extend(
        [
            "",
            "## Archived runs analyzed",
            "",
            f"Total runs with usable `final_state.json`: **{len(run_rows)}**",
        ]
    )

    if comparison_rows:
        lines.extend(
            [
                "",
                "## Vanilla vs skrub-full",
                "",
            ]
        )
        lines.extend(_markdown_table(comparison_rows, COMPARISON_TABLE_COLUMNS))
        lines.extend(_column_legend(COMPARISON_TABLE_COLUMNS))
        skrub_wins = sum(1 for r in comparison_rows if r["winner"] == "skrub-full")
        lines.append("")
        lines.append(
            f"Skrub-full win rate: **{skrub_wins}/{len(comparison_rows)}** tasks "
            f"({100 * skrub_wins / len(comparison_rows):.1f}%)."
        )

    if summary_rows:
        lines.extend(
            [
                "",
                "## Per (task, system) summary",
                "",
            ]
        )
        lines.extend(_markdown_table(summary_rows, SUMMARY_TABLE_COLUMNS))
        lines.extend(_column_legend(SUMMARY_TABLE_COLUMNS))

    if run_rows:
        lines.extend(
            [
                "",
                "## Per-run results",
                "",
            ]
        )
        lines.extend(
            _markdown_table(
                _all_runs_display_rows(run_rows),
                ALL_RUNS_TABLE_COLUMNS,
                decimals=4,
            )
        )
        lines.extend(_column_legend(ALL_RUNS_TABLE_COLUMNS))

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_results_bundle(
    config: EvalConfig,
    *,
    task_status: dict[str, Any],
    run_log: list[dict[str, Any]] | None,
    run_rows: list[dict[str, Any]],
) -> Path:
    stamp = config.runs_root.name if config.runs_root.name != "runs" else _utc_stamp()
    out_dir = config.results_dir / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = summarize_by_task_system(run_rows)
    comparison_rows = compare_systems(summary_rows)

    bundle = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "config": {
            "runs_root": str(config.runs_root.relative_to(REPO_ROOT)),
            "tasks_dir": str(TASKS_DIR.relative_to(REPO_ROOT)),
            "seed": config.seed,
            "model_label": config.model_label,
            "systems": config.systems,
            "repeats": config.repeats,
        },
        "task_status": task_status,
        "run_matrix": run_log,
        "run_count": len(run_rows),
        "comparisons": comparison_rows,
        "summary_by_task_system": summary_rows,
        "all_runs": run_rows,
    }
    (out_dir / "evaluation_summary.json").write_text(
        json.dumps(bundle, indent=2),
        encoding="utf-8",
    )
    (out_dir / "task_status.json").write_text(
        json.dumps(task_status, indent=2),
        encoding="utf-8",
    )
    if run_log is not None:
        (out_dir / "run_matrix.json").write_text(
            json.dumps(run_log, indent=2),
            encoding="utf-8",
        )

    write_markdown_report(
        out_dir / "report.md",
        config=config,
        task_status=task_status,
        run_rows=run_rows,
        summary_rows=summary_rows,
        comparison_rows=comparison_rows,
        run_log=run_log,
    )

    config.results_dir.mkdir(parents=True, exist_ok=True)
    (config.results_dir / "latest.json").write_text(
        json.dumps(
            {
                "latest_results_dir": str(out_dir.relative_to(REPO_ROOT)),
                "latest_runs_dir": str(config.runs_root.relative_to(REPO_ROOT)),
                "generated_at_utc": bundle["generated_at_utc"],
                "run_count": len(run_rows),
                "task_complete_count": task_status["complete_count"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_dir


def resolve_repeats(args: argparse.Namespace) -> list[str]:
    if args.repeat_count is not None:
        if args.repeat_count < 1:
            raise SystemExit("--repeat-count must be >= 1")
        return [f"run{i}" for i in range(1, args.repeat_count + 1)]
    return args.repeats


def parse_config(args: argparse.Namespace, *, eval_stamp: str) -> EvalConfig:
    runs_root = args.runs_root
    if runs_root is None:
        runs_root = RUNS_DIR / eval_stamp
    return EvalConfig(
        tasks=args.tasks or [],
        systems=args.systems,
        repeats=resolve_repeats(args),
        seed=args.seed,
        model_label=args.model_label,
        improved_agent_dir=args.improved_agent_dir.resolve(),
        vanilla_agent_dir=args.vanilla_agent_dir.resolve(),
        runs_root=runs_root.resolve(),
        results_dir=args.results_dir.resolve(),
        sync_tasks_to_vanilla=args.sync_tasks_to_vanilla,
        uv_sync_before_run=args.uv_sync_before_run,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
        summarize_only=getattr(args, "summarize_only", False),
    )


def build_run_parser(*, description: str | None = None) -> argparse.ArgumentParser:
    manifest = load_manifest(MANIFEST_PATH) if MANIFEST_PATH.is_file() else {}
    default_model = os.environ.get("ROOT_AGENT_MODEL", "openai/gpt-5.4-mini")
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Subset of task folder names to evaluate (default: all scanned tasks)",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["improved", "vanilla"],
        choices=sorted(SYSTEM_SPECS),
        help="Agent variants to run",
    )
    parser.add_argument(
        "--repeats",
        nargs="+",
        default=["run1"],
        help="Repeat labels for each (task, system) pair (ignored when --repeat-count is set)",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=None,
        help="Number of repeats; generates run1..runN (overrides --repeats)",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Directory for archived run bundles (default: automated_evaluation/runs/<stamp>)",
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--seed",
        type=int,
        default=manifest.get("default_seed", 42),
    )
    parser.add_argument("--model-label", type=str, default=default_model)
    parser.add_argument(
        "--improved-agent-dir",
        type=Path,
        default=IMPROVED_AGENT_DIR,
        help="Path to improved agent checkout",
    )
    parser.add_argument(
        "--vanilla-agent-dir",
        type=Path,
        default=DEFAULT_VANILLA_AGENT_DIR,
        help="Path to vanilla baseline agent checkout",
    )
    parser.add_argument(
        "--sync-tasks-to-vanilla",
        action="store_true",
        help="Copy scanned task packs into each agent checkout before running",
    )
    parser.add_argument(
        "--uv-sync-before-run",
        action="store_true",
        help="Run `uv sync` in the agent checkout before each run (slow; default keeps env warm)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the evaluation matrix without executing agents",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose archive already completed successfully (final_state.json + ADK prompt return in log)",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = build_run_parser()
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Skip agent execution; analyze an existing runs root and write results",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    eval_stamp = _utc_stamp()
    config = parse_config(args, eval_stamp=eval_stamp)

    tasks = scan_tasks()
    if config.tasks:
        wanted = set(config.tasks)
        tasks = [task for task in tasks if task.task_name in wanted]
    task_status = task_status_payload(tasks)

    run_log: list[dict[str, Any]] | None = None
    if not config.summarize_only:
        config.runs_root.mkdir(parents=True, exist_ok=True)
        run_log = execute_evaluation_matrix(config)
        if config.dry_run:
            print(f"\nDry run complete. Planned {len(run_log)} run(s).")
            return

    run_rows: list[dict[str, Any]] = []
    run_dirs = discover_run_dirs(config.runs_root)
    if not run_dirs:
        print(
            f"No archived runs found under {config.runs_root}. "
            "Writing task scan status only.",
            file=sys.stderr,
        )
    else:
        run_rows = analyze_runs(run_dirs, config.runs_root)
        print(f"Analyzed {len(run_rows)} run(s) from {config.runs_root}")

    out_dir = write_results_bundle(
        config,
        task_status=task_status,
        run_log=run_log,
        run_rows=run_rows,
    )
    print(f"\nResults written to: {out_dir}")
    print(f"Latest pointer: {config.results_dir / 'latest.json'}")


if __name__ == "__main__":
    main()
