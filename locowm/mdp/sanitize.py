"""Protect observations, rewards and world-model targets from non-finite physics data.

Sanitize each observation before noise and scaling. The default bound is
permissive; configuration overrides can impose tighter per-term bounds."""

from __future__ import annotations

import torch
from isaaclab.managers import ObservationTermCfg
from isaaclab.utils.modifiers import ModifierCfg

# Reject NaN/Inf and extreme magnitudes without clipping normal physical values.
DEFAULT_CLIP: float = 1.0e6


def sanitize(x: torch.Tensor, clip: float = DEFAULT_CLIP) -> torch.Tensor:
    """Replace NaN/Inf with finite values and clamp to [-clip, clip]."""
    return torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip).clamp(-clip, clip)


def obs_sanitize_modifier(data: torch.Tensor, clip: float = DEFAULT_CLIP) -> torch.Tensor:
    """Sanitize an observation term before noise and scaling.

    Keep explicit named arguments: Isaac Lab validates the modifier signature
    and does not accept arbitrary **kwargs."""
    return sanitize(data, clip)


def add_obs_sanitizers(
    obs_cfg,
    default_clip: float = DEFAULT_CLIP,
    overrides: dict[str, float] | None = None,
) -> None:
    """Prepend a sanitizer to each observation term, skipping terms already configured.

    Overrides map term names to pre-scale clipping bounds. Calling this again
    after a subclass adds observations is safe."""
    overrides = overrides or {}
    for group in vars(obs_cfg).values():
        if group is None:
            continue
        for name, term in vars(group).items():
            if not isinstance(term, ObservationTermCfg):
                continue
            mods = list(term.modifiers or [])
            if any(getattr(m, "func", None) is obs_sanitize_modifier for m in mods):
                continue
            clip = overrides.get(name, default_clip)
            term.modifiers = [ModifierCfg(func=obs_sanitize_modifier, params={"clip": clip})] + mods
