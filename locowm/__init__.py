"""LocoWM Isaac Lab extension registration.

Importing the lightweight path helpers should remain possible on machines
without the Omniverse runtime. Simulation entry points call
``ensure_runtime`` after launching Isaac Sim so a missing runtime still
produces an actionable error before task creation.
"""

from __future__ import annotations

_BLACKLIST_PKGS = ["utils"]
_RUNTIME_ERROR: ModuleNotFoundError | None = None
_REGISTERED = False


def ensure_runtime() -> None:
    """Load Isaac Lab and register tasks after ``SimulationApp`` is started."""

    global _RUNTIME_ERROR, _REGISTERED

    if _REGISTERED:
        return

    if _RUNTIME_ERROR is not None:
        raise RuntimeError(
            "LocoWM requires the supported Isaac Sim 5.1.0 / Isaac Lab environment. "
            "The Omniverse runtime could not be imported. Install Isaac Sim and Isaac Lab "
            "in the same environment before launching a simulation command."
        ) from _RUNTIME_ERROR

    try:
        from isaaclab_tasks.utils import import_packages
        import_packages(__name__, _BLACKLIST_PKGS)
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on Isaac installation
        _RUNTIME_ERROR = exc
        raise RuntimeError(
            "LocoWM requires the supported Isaac Sim 5.1.0 / Isaac Lab environment. "
            "The Omniverse runtime could not be imported. Install Isaac Sim and Isaac Lab "
            "in the same environment before launching a simulation command."
        ) from exc
    _REGISTERED = True
