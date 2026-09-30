"""Runtime path helpers shared by the public command-line entry points.

The Isaac Lab launch scripts are often run from a directory other than the
repository root.  Keeping path handling here makes the experiment commands
independent of the caller's current working directory while leaving Isaac
Lab's own configuration loading untouched.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = PROJECT_ROOT / "logs" / "rsl_rl"


def project_path(path: str | Path) -> Path:
    """Return an absolute path, resolving relative values from the repository."""

    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def existing_file(path: str | Path, *, label: str) -> str:
    """Resolve a required file and produce a useful error when it is missing."""

    candidate = project_path(path)
    if not candidate.is_file():
        raise FileNotFoundError(
            f"{label} not found: '{path}'. Resolved path: '{candidate}'. "
            "Pass a repository-relative path or an absolute path."
        )
    return str(candidate)


def experiment_log_root(experiment_name: str) -> str:
    """Return the absolute log root for an experiment."""

    return str(LOG_ROOT / experiment_name)


def output_path(path: str | Path | None, *parts: str) -> str:
    """Resolve an optional output directory, defaulting below the repository."""

    if path is not None:
        return str(project_path(path))
    return str(PROJECT_ROOT.joinpath(*parts))
