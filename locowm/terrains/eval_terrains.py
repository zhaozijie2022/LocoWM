from __future__ import annotations

import numpy as np
import scipy.interpolate as interpolate
from typing import TYPE_CHECKING

from isaaclab.terrains.height_field.utils import height_field_to_mesh

from .custom_terrains import _perlin_like_noise_2d

if TYPE_CHECKING:
    from . import custom_terrains_cfg


# =============================================================================
# Evaluation terrains.
#

# Geometry comes from explicit configuration fields; ignore generator difficulty.
# eval_env_cfg scales these fields for each evaluation tier.
# The robot starts near the cell center and moves along positive x.
# Keep compatible cell boundaries for tiling.
# Height-field axis 0 is forward x; axis 1 is lateral y.
# =============================================================================


@height_field_to_mesh
def eval_slope_hill_terrain(difficulty: float, cfg: "custom_terrains_cfg.HfEvalSlopeHillTerrainCfg") -> np.ndarray:
    """Build an uphill/plateau/downhill ridge in the positive-x half.

    The negative-x half and cell edges stay flat. Explicit slope_tan sets the
    slope; the generator difficulty is ignored."""
    hs = cfg.horizontal_scale
    width_px = int(cfg.size[0] / hs)  # Forward x.
    length_px = int(cfg.size[1] / hs)  # Lateral y.
    hf = np.zeros((width_px, length_px), dtype=np.int_)

    center_px = width_px // 2
    pre_flat_px = int(cfg.pre_flat / hs)
    ramp_run_px = max(1, int(cfg.ramp_run / hs))
    top_len_px = int(cfg.top_len / hs)

    # Height increment per horizontal pixel, in vertical-scale units.
    slope_per_px = cfg.slope_tan * hs / cfg.vertical_scale

    x0 = center_px + pre_flat_px  # Start of the uphill segment.
    x1 = x0 + ramp_run_px  # Start of the plateau.
    x2 = x1 + top_len_px  # Start of the downhill segment.
    x3 = x2 + ramp_run_px  # Return to zero height.
    peak = slope_per_px * ramp_run_px

    for i in range(width_px):
        if x0 <= i < x1:
            h = slope_per_px * (i - x0)
        elif x1 <= i < x2:
            h = peak
        elif x2 <= i < x3:
            h = peak - slope_per_px * (i - x2)
        else:
            h = 0.0
        hf[i, :] = int(round(h))
    return hf


@height_field_to_mesh
def eval_vertical_bump_terrain(
    difficulty: float, cfg: "custom_terrains_cfg.HfEvalVerticalBumpTerrainCfg"
) -> np.ndarray:
    """Build periodic rectangular bumps with a flat center and cell-edge margins.

    The slope threshold produces vertical walls. Geometry is set explicitly;
    the generator difficulty is ignored."""
    hs = cfg.horizontal_scale
    width_px = int(cfg.size[0] / hs)
    length_px = int(cfg.size[1] / hs)
    hf = np.zeros((width_px, length_px), dtype=np.int_)

    h_px = int(round(cfg.bump_height / cfg.vertical_scale))
    w_px = max(1, int(round(cfg.bump_width / hs)))
    period_px = max(w_px + 1, int(round(cfg.bump_period / hs)))
    margin_px = int(round(cfg.edge_margin / hs))
    center_px = width_px // 2

    # Tile bumps symmetrically around a flat central gap.
    centers: list[int] = []
    k = 0
    while True:
        xc = center_px + int(period_px * (k + 0.5))
        if xc + w_px // 2 > width_px - margin_px:
            break
        centers.append(xc)
        k += 1
    k = 0
    while True:
        xc = center_px - int(period_px * (k + 0.5))
        if xc - w_px // 2 < margin_px:
            break
        centers.append(xc)
        k += 1

    for xc in centers:
        a = max(0, xc - w_px // 2)
        b = min(width_px, xc + w_px // 2)
        hf[a:b, :] = h_px
    return hf


@height_field_to_mesh
def eval_perlin_terrain(difficulty: float, cfg: "custom_terrains_cfg.HfEvalPerlinTerrainCfg") -> np.ndarray:
    """Build periodic Perlin noise with fixed peak-to-peak amplitude and minimum zero.

    Reuse the periodic noise kernel and spline upsampling of custom terrain.
    The generator difficulty is ignored."""
    hs = cfg.horizontal_scale
    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = hs
    elif cfg.downsampled_scale < hs:
        raise ValueError(f"Downsampled scale must be >= horizontal scale: {cfg.downsampled_scale} < {hs}.")

    width_pixels = int(cfg.size[0] / hs)
    length_pixels = int(cfg.size[1] / hs)
    width_ds = int(cfg.size[0] / cfg.downsampled_scale)
    length_ds = int(cfg.size[1] / cfg.downsampled_scale)
    amp_px = cfg.amplitude / cfg.vertical_scale

    n = _perlin_like_noise_2d(
        (width_ds, length_ds),
        frequency=cfg.frequency,
        octaves=cfg.octaves,
        persistence=cfg.persistence,
        lacunarity=cfg.lacunarity,
        seed=cfg.seed,
        repeat=(width_ds, length_ds),
    )
    # n in [-1, 1] -> [0, amplitude]
    hf_ds = (n + 1.0) * 0.5 * amp_px

    # Match the spline upsampling coordinates used by custom Perlin terrain.
    x = np.linspace(0, cfg.size[0] * hs, width_ds)
    y = np.linspace(0, cfg.size[1] * hs, length_ds)
    func = interpolate.RectBivariateSpline(x, y, hf_ds)
    x_up = np.linspace(0, cfg.size[0] * hs, width_pixels)
    y_up = np.linspace(0, cfg.size[1] * hs, length_pixels)
    z_up = func(x_up, y_up)
    z = np.rint(z_up).astype(np.int16)

    return z


@height_field_to_mesh
def eval_unilateral_bridge_terrain(
    difficulty: float, cfg: "custom_terrains_cfg.HfEvalUnilateralBridgeTerrainCfg"
) -> np.ndarray:
    """Raise the positive-y half to bridge_height, constant along x.

    One side of the robot travels on the bridge and the other on the ground.
    The generator difficulty is ignored."""
    hs = cfg.horizontal_scale
    width_px = int(cfg.size[0] / hs)
    length_px = int(cfg.size[1] / hs)
    hf = np.zeros((width_px, length_px), dtype=np.int_)

    h_px = int(round(cfg.bridge_height / cfg.vertical_scale))
    center_y = length_px // 2
    hf[:, center_y:] = h_px  # Raise the positive-y half.
    return hf


# =============================================================================
# Success-evaluation terrains contain one bounded feature band.
#
# Layout along positive x, starting at the cell center:
# Flat entry section of length pre_flat.
# Feature band of length feat_len.
# Flat exit section to the cell boundary.
# The negative-x half stays flat; a single row prevents a second feature traversal.
# Slopes reuse eval_slope_hill_terrain, which already has a bounded ridge.
# =============================================================================


def _succ_band(cfg):
    """Return width, length, center, band-start and band-end pixel indices."""
    hs = cfg.horizontal_scale
    width_px = int(cfg.size[0] / hs)
    length_px = int(cfg.size[1] / hs)
    center_px = width_px // 2
    band0 = center_px + int(cfg.pre_flat / hs)
    band1 = band0 + int(cfg.feat_len / hs)
    return width_px, length_px, center_px, band0, band1


@height_field_to_mesh
def succ_bump_terrain(difficulty: float, cfg: "custom_terrains_cfg.HfSuccBumpTerrainCfg") -> np.ndarray:
    """Create rectangular bumps within one feature band, ignoring difficulty."""
    hs = cfg.horizontal_scale
    width_px, length_px, _c, band0, band1 = _succ_band(cfg)
    hf = np.zeros((width_px, length_px), dtype=np.int_)

    h_px = int(round(cfg.bump_height / cfg.vertical_scale))
    w_px = max(1, int(round(cfg.bump_width / hs)))
    period_px = max(w_px + 1, int(round(cfg.bump_period / hs)))
    xc = band0 + period_px // 2
    while xc + w_px // 2 <= band1:
        hf[xc - w_px // 2 : xc + w_px // 2, :] = h_px
        xc += period_px
    return hf


@height_field_to_mesh
def succ_perlin_terrain(difficulty: float, cfg: "custom_terrains_cfg.HfSuccPerlinTerrainCfg") -> np.ndarray:
    """Create fixed-amplitude Perlin noise inside the feature band and flat ground outside."""
    hs = cfg.horizontal_scale
    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = hs
    width_pixels, length_pixels, _c, band0, band1 = _succ_band(cfg)
    width_ds = int(cfg.size[0] / cfg.downsampled_scale)
    length_ds = int(cfg.size[1] / cfg.downsampled_scale)
    amp_px = cfg.amplitude / cfg.vertical_scale

    n = _perlin_like_noise_2d(
        (width_ds, length_ds),
        frequency=cfg.frequency,
        octaves=cfg.octaves,
        persistence=cfg.persistence,
        lacunarity=cfg.lacunarity,
        seed=cfg.seed,
        repeat=(width_ds, length_ds),
    )
    hf_ds = (n + 1.0) * 0.5 * amp_px
    x = np.linspace(0, cfg.size[0] * hs, width_ds)
    y = np.linspace(0, cfg.size[1] * hs, length_ds)
    func = interpolate.RectBivariateSpline(x, y, hf_ds)
    x_up = np.linspace(0, cfg.size[0] * hs, width_pixels)
    y_up = np.linspace(0, cfg.size[1] * hs, length_pixels)
    z = np.rint(func(x_up, y_up)).astype(np.int16)

    # Keep the region outside the feature band flat.
    z[:band0, :] = 0
    z[band1:, :] = 0
    return z


@height_field_to_mesh
def succ_bridge_terrain(difficulty: float, cfg: "custom_terrains_cfg.HfSuccBridgeTerrainCfg") -> np.ndarray:
    """Raise the positive-y half within the feature band, with flat entry and exit sections."""

    width_px, length_px, _c, band0, band1 = _succ_band(cfg)
    hf = np.zeros((width_px, length_px), dtype=np.int_)

    h_px = int(round(cfg.bridge_height / cfg.vertical_scale))
    cy = length_px // 2
    hf[band0:band1, cy:] = h_px
    return hf
