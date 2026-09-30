from __future__ import annotations

import numpy as np
import scipy.interpolate as interpolate
from scipy.ndimage import zoom
from typing import TYPE_CHECKING

from isaaclab.terrains.height_field.utils import height_field_to_mesh

if TYPE_CHECKING:
    from . import custom_terrains_cfg


def _perlin_like_noise_2d(
    shape: tuple[int, int],
    frequency: float,
    octaves: int,
    persistence: float,
    lacunarity: float,
    seed: int,
    repeat: tuple[int, int] | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros(shape, dtype=np.float64)
    amp = 1.0
    freq_scale = 1.0
    amp_sum = 0.0
    for _ in range(octaves):
        gw = max(2, int(shape[0] * frequency * freq_scale))
        gl = max(2, int(shape[1] * frequency * freq_scale))
        if repeat is not None:
            gw = min(gw, repeat[0])
            gl = min(gl, repeat[1])
        gw = max(2, gw)
        gl = max(2, gl)
        grid = rng.uniform(-1.0, 1.0, (gw + 1, gl + 1))
        if repeat is not None and (repeat[0] > 0 and repeat[1] > 0):
            grid[-1, :] = grid[0, :]
            grid[:, -1] = grid[:, 0]
        zoom_factors = (shape[0] / grid.shape[0], shape[1] / grid.shape[1])
        layer = zoom(grid, zoom_factors, order=1, mode="wrap" if repeat else "nearest")
        out += amp * layer
        amp_sum += amp
        amp *= persistence
        freq_scale *= lacunarity
    if amp_sum > 0:
        out /= amp_sum
    return np.clip(out, -1.0, 1.0)


@height_field_to_mesh
def perlin_noise_terrain(difficulty: float, cfg: custom_terrains_cfg.HfPerlinNoiseTerrainCfg) -> np.ndarray:
    # check parameters
    # -- horizontal scale
    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = cfg.horizontal_scale
    elif cfg.downsampled_scale < cfg.horizontal_scale:
        raise ValueError(
            "Downsampled scale must be larger than or equal to the horizontal scale:"
            f" {cfg.downsampled_scale} < {cfg.horizontal_scale}."
        )

    # switch parameters to discrete units
    # -- horizontal scale
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- downsampled scale
    width_downsampled = int(cfg.size[0] / cfg.downsampled_scale)
    length_downsampled = int(cfg.size[1] / cfg.downsampled_scale)
    # -- height
    height_min = int(cfg.noise_range[0] / cfg.vertical_scale)
    height_max = int(((cfg.noise_range[1] - cfg.noise_range[0]) * difficulty + cfg.noise_range[0]) / cfg.vertical_scale)

    n = _perlin_like_noise_2d(
        (width_downsampled, length_downsampled),
        frequency=cfg.frequency,
        octaves=cfg.octaves,
        persistence=cfg.persistence,
        lacunarity=cfg.lacunarity,
        seed=cfg.seed,
        repeat=(width_downsampled, length_downsampled),
    )
    height_field_downsampled = height_min + (n + 1.0) * 0.5 * (height_max - height_min)
    # create interpolation function for the sampled heights

    x = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_downsampled)
    y = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_downsampled)
    func = interpolate.RectBivariateSpline(x, y, height_field_downsampled)

    # interpolate the sampled heights to obtain the height field
    x_upsampled = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_pixels)
    y_upsampled = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_pixels)
    z_upsampled = func(x_upsampled, y_upsampled)
    # round off the interpolated heights to the nearest vertical step
    return np.rint(z_upsampled).astype(np.int16)


@height_field_to_mesh
def custom_random_uniform_terrain(
    difficulty: float, cfg: custom_terrains_cfg.HfCustomRandomUniformTerrainCfg
) -> np.ndarray:
    """Generate uniform height noise with a difficulty-dependent upper bound.

    The upper bound is min_height + (max_height - min_height) * difficulty.
    Sampling and interpolation otherwise follow Isaac Lab random uniform terrain."""
    # check parameters
    # -- horizontal scale
    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = cfg.horizontal_scale
    elif cfg.downsampled_scale < cfg.horizontal_scale:
        raise ValueError(
            "Downsampled scale must be larger than or equal to the horizontal scale:"
            f" {cfg.downsampled_scale} < {cfg.horizontal_scale}."
        )

    # switch parameters to discrete units
    # -- horizontal scale
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- downsampled scale
    width_downsampled = int(cfg.size[0] / cfg.downsampled_scale)
    length_downsampled = int(cfg.size[1] / cfg.downsampled_scale)
    # -- height
    height_min = int(cfg.noise_range[0] / cfg.vertical_scale)
    height_max = int(((cfg.noise_range[1] - cfg.noise_range[0]) * difficulty + cfg.noise_range[0]) / cfg.vertical_scale)
    height_step = int(cfg.noise_step / cfg.vertical_scale)

    # create range of heights possible
    height_range = np.arange(height_min, height_max + height_step, height_step)
    # sample heights randomly from the range along a grid
    height_field_downsampled = np.random.choice(height_range, size=(width_downsampled, length_downsampled))
    # create interpolation function for the sampled heights
    x = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_downsampled)
    y = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_downsampled)
    func = interpolate.RectBivariateSpline(x, y, height_field_downsampled)

    # interpolate the sampled heights to obtain the height field
    x_upsampled = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_pixels)
    y_upsampled = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_pixels)
    z_upsampled = func(x_upsampled, y_upsampled)
    # round off the interpolated heights to the nearest vertical step
    return np.rint(z_upsampled).astype(np.int16)


@height_field_to_mesh
def speed_bump_terrain(difficulty: float, cfg: custom_terrains_cfg.HfSpeedBumpTerrainCfg) -> np.ndarray:
    width_px = int(cfg.size[0] / cfg.horizontal_scale)
    length_px = int(cfg.size[1] / cfg.horizontal_scale)
    hf = np.zeros((width_px, length_px), dtype=np.int_)

    actual_height = cfg.bump_height_range[0] + difficulty * (cfg.bump_height_range[1] - cfg.bump_height_range[0])
    bump_height_px = int(actual_height / cfg.vertical_scale)

    # Start of the uphill segment for each bump.
    x_starts_px = np.arange(width_px / cfg.num_bumps / 2, width_px, width_px / cfg.num_bumps).astype(dtype=np.int_)
    # Sample the bump geometry.
    bump_width = np.random.uniform(cfg.random_bump_width[0], cfg.random_bump_width[1], size=cfg.num_bumps)
    flat_ratio = np.random.uniform(cfg.random_flat_ratio[0], cfg.random_flat_ratio[1], size=cfg.num_bumps)
    flat_width = bump_width * flat_ratio
    ramp_width = (bump_width - flat_width) / 2
    ramp_width_px = np.round(ramp_width / cfg.horizontal_scale).astype(np.int_)
    flat_width_px = np.round(flat_width / cfg.horizontal_scale).astype(np.int_)
    bump_width_px = ramp_width_px * 2 + flat_width_px

    # Construct periodic bumps.
    for i in range(width_px):
        h = 0
        for k in range(cfg.num_bumps):
            dx_px = i - x_starts_px[k]
            if 0 <= dx_px < bump_width_px[k]:
                if dx_px < ramp_width_px[k]:
                    h = bump_height_px / ramp_width_px[k] * dx_px
                elif dx_px < ramp_width_px[k] + flat_width_px[k]:
                    h = bump_height_px
                else:
                    h = bump_height_px / ramp_width_px[k] * (bump_width_px[k] - dx_px)
                break
        hf[i, :] = int(round(h))

    # Reserve a flat central platform for spawning the robot.
    platform_radius_px = int(cfg.platform_width / 2 / cfg.horizontal_scale)
    center_x_px = width_px // 2
    center_y_px = length_px // 2
    hf[
        center_x_px - platform_radius_px : center_x_px + platform_radius_px,
        center_y_px - platform_radius_px : center_y_px + platform_radius_px,
    ] = 0

    # Cut random gaps along the lateral direction within each bump.
    y_periods = np.linspace(cfg.gap_margin, (length_px - cfg.gap_margin), cfg.num_gaps + 1)
    y_periods_px = np.round(y_periods).astype(np.int_)

    for k in range(cfg.num_bumps):
        gap_length = np.random.uniform(cfg.random_gap_length[0], cfg.random_gap_length[1], size=cfg.num_gaps)
        gap_length_px = np.round(gap_length / cfg.horizontal_scale).astype(np.int_)

        for p in range(cfg.num_gaps):
            y0 = int(np.random.uniform(y_periods_px[p], y_periods_px[p + 1] - gap_length_px[p]))
            y1 = y0 + gap_length_px[p]

            hf[x_starts_px[k] : x_starts_px[k] + bump_width_px[k], y0:y1] = 0

    return hf


@height_field_to_mesh
def x_wave_terrain(difficulty: float, cfg: custom_terrains_cfg.HfXWaveTerrainCfg) -> np.ndarray:
    r"""
    h(x, y) =  A \left(\sin\left(\frac{2 \pi x}{\lambda}\right)
    """
    if isinstance(cfg.wave_length, tuple):
        wave_length = cfg.wave_length[0] + difficulty * (cfg.wave_length[1] - cfg.wave_length[0])
    else:
        wave_length = cfg.wave_length

    if wave_length <= 0:
        raise ValueError(f"wave_length must be positive. Got: {wave_length}")

    width_px = int(cfg.size[0] / cfg.horizontal_scale)
    length_px = int(cfg.size[1] / cfg.horizontal_scale)

    amplitude = cfg.amplitude_range[0] + difficulty * (cfg.amplitude_range[1] - cfg.amplitude_range[0])

    x = np.linspace(0, cfg.size[0], width_px)
    y = np.linspace(0, cfg.size[1], length_px)
    xv, yv = np.meshgrid(x, y, indexing="ij")

    h_meters = amplitude * np.sin(2.0 * np.pi * xv / wave_length)
    hf = np.round(h_meters / cfg.vertical_scale).astype(np.int16)

    return hf
