from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from . import custom_terrains
from . import eval_terrains


@configclass
class HfCustomRandomUniformTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a random uniform height field terrain."""

    function = custom_terrains.custom_random_uniform_terrain

    noise_range: tuple[float, float] = MISSING
    """The minimum and maximum height noise (i.e. along z) of the terrain (in m)."""
    noise_step: float = MISSING
    """The minimum height (in m) change between two points."""
    downsampled_scale: float | None = None
    """The distance between two randomly sampled points on the terrain. Defaults to None,
    in which case the :obj:`horizontal scale` is used.

    The heights are sampled at this resolution and interpolation is performed for intermediate points.
    This must be larger than or equal to the :obj:`horizontal scale`.
    """


@configclass
class HfPerlinNoiseTerrainCfg(HfTerrainBaseCfg):
    function = custom_terrains.perlin_noise_terrain

    frequency: float = 0.1
    octaves: int = 4
    lacunarity: float = 2.0
    persistence: float = 0.5
    seed: int = 42

    noise_range: tuple[float, float] = MISSING
    """The minimum and maximum height noise (i.e. along z) of the terrain (in m)."""
    noise_step: float = MISSING
    """The minimum height (in m) change between two points."""
    downsampled_scale: float | None = None
    """The distance between two randomly sampled points on the terrain. Defaults to None,
    in which case the :obj:`horizontal scale` is used.

    The heights are sampled at this resolution and interpolation is performed for intermediate points.
    This must be larger than or equal to the :obj:`horizontal scale`.
    """


@configclass
class HfSpeedBumpTerrainCfg(HfTerrainBaseCfg):
    """Periodic triangular or trapezoidal speed bumps."""

    function = custom_terrains.speed_bump_terrain

    num_bumps: int = 8
    """Number of speed bumps."""

    bump_height_range: tuple[float, float] = MISSING
    """Bump-height range in meters, scaled by difficulty."""

    random_flat_ratio: tuple[float, float] = MISSING
    """Random plateau fraction in [0, 1], independent of difficulty."""

    random_bump_width: tuple[float, float] = MISSING
    """Random bump-width range in meters, independent of difficulty."""

    num_gaps: int = 4
    """Number of gaps per bump."""

    random_gap_length: tuple[float, float] = MISSING
    """Random gap-length range in meters, independent of difficulty."""

    gap_margin: float = 0.5
    """Margin around bump gaps (m)."""

    platform_width: float = 2.0
    """Width of the central flat platform (m)."""


@configclass
class HfXWaveTerrainCfg(HfTerrainBaseCfg):
    """Waves along the x axis with evenly spaced wavelengths."""

    function = custom_terrains.x_wave_terrain

    amplitude_range: tuple[float, float] = MISSING
    """The minimum and maximum amplitude of the wave (in m)."""

    wave_length: float | tuple[float, float] = MISSING


# =============================================================================
# Evaluation terrain configurations.
# Explicit geometry overrides the generator difficulty argument.
# eval_env_cfg scales these fields by difficulty tier.
# =============================================================================


@configclass
class HfEvalSlopeHillTerrainCfg(HfTerrainBaseCfg):
    """Evaluation ridge with uphill, plateau and downhill segments."""

    function = eval_terrains.eval_slope_hill_terrain

    slope_tan: float = 0.30
    """Tangent of the slope angle; 0.30 at tier 100."""
    pre_flat: float = 1.0
    """Flat entry length before the ridge in the positive-x half (m)."""
    ramp_run: float = 1.0
    """Horizontal length of each ramp (m); peak height is slope_tan * ramp_run."""
    top_len: float = 0.4
    """Plateau length (m)."""


@configclass
class HfEvalVerticalBumpTerrainCfg(HfTerrainBaseCfg):
    """Periodic rectangular bumps with vertical walls."""

    function = eval_terrains.eval_vertical_bump_terrain

    bump_height: float = 0.12
    """Bump height at tier 100 (m)."""
    bump_width: float = 0.20
    """Forward width of each bump (m)."""
    bump_period: float = 1.0
    """Bump period (m)."""
    edge_margin: float = 0.30
    """Flat margin at each cell boundary for tiling (m)."""


@configclass
class HfEvalPerlinTerrainCfg(HfTerrainBaseCfg):
    """Periodic Perlin noise with a fixed peak-to-peak amplitude."""

    function = eval_terrains.eval_perlin_terrain

    amplitude: float = 0.10
    """Peak-to-peak noise amplitude at tier 100 (m)."""
    frequency: float = 0.75
    octaves: int = 2
    lacunarity: float = 2.0
    persistence: float = 0.5
    seed: int = 42
    downsampled_scale: float | None = None
    platform_radius: float = 0.8
    """Reserved central platform radius (m); unused by the current generator."""


@configclass
class HfEvalUnilateralBridgeTerrainCfg(HfTerrainBaseCfg):
    """Raise the positive-y half to a constant bridge height."""

    function = eval_terrains.eval_unilateral_bridge_terrain

    bridge_height: float = 0.25
    """Bridge height above the ground at tier 100 (m)."""


# ---------------------------------------------------------------------------
# Success-evaluation configurations contain one bounded feature band.
# pre_flat sets the entry length; feat_len sets the feature-band length.
# Slopes reuse HfEvalSlopeHillTerrainCfg with bounded ramp dimensions.
# ---------------------------------------------------------------------------


@configclass
class HfSuccBumpTerrainCfg(HfTerrainBaseCfg):
    """Rectangular bumps restricted to one bounded feature band."""

    function = eval_terrains.succ_bump_terrain

    pre_flat: float = 2.0
    feat_len: float = 2.5
    bump_height: float = 0.12
    bump_width: float = 0.20
    bump_period: float = 1.0


@configclass
class HfSuccPerlinTerrainCfg(HfTerrainBaseCfg):
    """Fixed-amplitude Perlin noise within a bounded feature band."""

    function = eval_terrains.succ_perlin_terrain

    pre_flat: float = 2.0
    feat_len: float = 2.5
    amplitude: float = 0.10
    frequency: float = 0.75
    octaves: int = 2
    lacunarity: float = 2.0
    persistence: float = 0.5
    seed: int = 42
    downsampled_scale: float | None = None


@configclass
class HfSuccBridgeTerrainCfg(HfTerrainBaseCfg):
    """A bounded unilateral bridge with flat entry and exit sections."""

    function = eval_terrains.succ_bridge_terrain

    pre_flat: float = 2.0
    feat_len: float = 2.5
    bridge_height: float = 0.25
