"""Evaluate control and payload metrics over four terrains and four difficulty tiers.

A fixed forward-speed ramp precedes the measurement window. Base, E2E,
Adapter and NoWM tasks retain the critic dimensions used during training.
Metrics include tracking error, posture, vertical motion, payload retention
and sliding. See README for checkpoint arguments and example commands."""

import argparse
import os
import sys

if __package__ in {None, ""}:  # Support the legacy ``python locowm/scripts/eval.py`` form.
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from isaaclab.app import AppLauncher
from locowm.scripts import cli_args
from locowm.scripts.runtime import experiment_log_root, output_path

parser = argparse.ArgumentParser(description="Evaluate an RSL-RL policy over multi-terrain + payload.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during eval.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments (best a multiple of 16).")
parser.add_argument("--task", type=str, default=None, help="Name of the eval task.")
# Evaluation protocol and measurement window.
parser.add_argument(
    "--target_speed", type=float, default=1.0, help="Forward velocity after the acceleration ramp (m/s)."
)
parser.add_argument("--accel_time", type=float, default=2.5, help="Time to ramp from zero to target_speed (s).")
parser.add_argument(
    "--settle_time", type=float, default=0.5, help="Settling time between acceleration and measurement (s)."
)
parser.add_argument("--measure_time", type=float, default=15.0, help="Measurement window duration (s).")
parser.add_argument(
    "--feature_gate_thresh",
    type=float,
    default=0.015,
    help="Minimum height-scan span (m) for tracking and posture metrics; excludes flat gaps.",
)
parser.add_argument(
    "--tip_angle_deg", type=float, default=45.0, help="Payload tilt threshold for detecting a tip (degrees)."
)
parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: logs/eval/<task>/<time>).")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import locowm

locowm.ensure_runtime()


import gymnasium as gym
import json
import math
from datetime import datetime

import torch

from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi, quat_apply_inverse
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from locowm.config.go2w.eval_env_cfg import EVAL_CONDITIONS, EVAL_TERRAIN_TYPES, EVAL_TIERS


def main():

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # Override the velocity-ramp parameters.
    step_dt = env_cfg.decimation * env_cfg.sim.dt
    accel_steps = max(1, round(args_cli.accel_time / step_dt))
    env_cfg.commands.base_velocity.target_speed = float(args_cli.target_speed)
    env_cfg.commands.base_velocity.accel_steps = int(accel_steps)

    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = experiment_log_root(agent_cfg.experiment_name)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "eval"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording eval video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    from loco_rl.runners import OnPolicyRunner

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs

    # Resolve each environment's terrain condition.
    # terrain_types indexes EVAL_CONDITIONS (0..15).
    cond = env.unwrapped.scene.terrain.terrain_types.clone().to(device).long()

    # Measurement window.
    warmup_steps = accel_steps + max(0, round(args_cli.settle_time / step_dt))
    measure_steps = max(1, round(args_cli.measure_time / step_dt))
    total_steps = warmup_steps + measure_steps
    tip_start = 10  # Allow the payload to settle before checking for tipping.
    gate_thr = float(args_cli.feature_gate_thresh)
    tip_thr = math.radians(float(args_cli.tip_angle_deg))

    print(
        f"[INFO] num_envs={num_envs}, step_dt={step_dt:.4f}s, warmup={warmup_steps}, "
        f"measure={measure_steps}, total={total_steps} steps",
        flush=True,
    )

    robot = env.unwrapped.scene["robot"]
    obj = env.unwrapped.scene["object"]
    scanner = env.unwrapped.scene.sensors["height_scanner"]

    # Per-environment accumulators.
    z = lambda: torch.zeros(num_envs, device=device)
    n_gate = z()  # Steps that are measured, on a terrain feature and still alive.
    sum_velerr2 = z()  # Σ(cmd_vx - vx)^2
    sum_pitch2 = z()
    sum_roll2 = z()
    sum_absvz = z()
    max_pitch = z()
    max_roll = z()
    max_absvz = z()
    tipped = torch.zeros(num_envs, dtype=torch.bool, device=device)
    slide_max = z()
    rel0_xy = torch.zeros(num_envs, 2, device=device)
    captured0 = torch.zeros(num_envs, dtype=torch.bool, device=device)
    alive = torch.ones(num_envs, dtype=torch.bool, device=device)

    obs, _ = env.get_observations()

    for step in range(total_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            dones = dones.bool()

            # Robot state.
            vx = robot.data.root_lin_vel_b[:, 0]
            vz_w = robot.data.root_lin_vel_w[:, 2]
            roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
            roll = wrap_to_pi(roll)
            pitch = wrap_to_pi(pitch)
            cmd_vx = env.unwrapped.command_manager.get_command("base_velocity")[:, 0]

            base_pos_w = robot.data.root_pos_w
            base_quat = robot.data.root_quat_w

            # Use the height-scan span to detect terrain features.
            hits_z = scanner.data.ray_hits_w[..., 2]
            hits_z = torch.nan_to_num(hits_z, nan=0.0, posinf=0.0, neginf=0.0)
            span = hits_z.amax(dim=1) - hits_z.amin(dim=1)

            # Payload state.
            obj_g = obj.data.projected_gravity_b
            tilt = torch.acos(torch.clamp(-obj_g[:, 2], -1.0, 1.0))  # Zero corresponds to upright.
            obj_pos_w = obj.data.root_pos_w
            rel = quat_apply_inverse(base_quat, obj_pos_w - base_pos_w)
            rel_xy = rel[:, :2]
            fell = obj_pos_w[:, 2] < (base_pos_w[:, 2] - 0.15)

            # Phase masks.
            measure_now = step >= warmup_steps
            active = alive & (~dones)  # bool (num_envs,)
            zeros_b = torch.zeros_like(active)
            gate = (active & (span > gate_thr)) if measure_now else zeros_b

            # Check tipping after settling, independently of the terrain-feature gate.
            if step >= tip_start:
                tipped |= active & ((tilt > tip_thr) | fell)

            # Record the initial payload offset at the start of the measurement window.
            if measure_now:
                need0 = active & (~captured0)
                rel0_xy[need0] = rel_xy[need0]
                captured0 |= need0

            # Accumulate gated tracking and posture metrics.
            gf = gate.float()
            n_gate += gf
            sum_velerr2 += gf * (cmd_vx - vx) ** 2
            sum_pitch2 += gf * pitch**2
            sum_roll2 += gf * roll**2
            sum_absvz += gf * vz_w.abs()
            max_pitch = torch.maximum(max_pitch, torch.where(gate, pitch.abs(), max_pitch))
            max_roll = torch.maximum(max_roll, torch.where(gate, roll.abs(), max_roll))
            max_absvz = torch.maximum(max_absvz, torch.where(gate, vz_w.abs(), max_absvz))

            # Track maximum horizontal displacement from the initial payload offset.
            slide_now = torch.linalg.norm(rel_xy - rel0_xy, dim=1)
            smask = (active & captured0) if measure_now else zeros_b
            slide_max = torch.maximum(slide_max, torch.where(smask, slide_now, slide_max))

            alive &= ~dones

    # Per-environment metrics.
    cnt = n_gate.clamp(min=1.0)
    ev_velmse = sum_velerr2 / cnt
    ev_pitchmse = sum_pitch2 / cnt
    ev_rollmse = sum_roll2 / cnt
    ev_absvz = sum_absvz / cnt
    has = n_gate > 0  # Environments with at least one valid measurement.

    # Aggregate means and standard deviations by condition.
    def _ms(t, mask):
        """Return masked mean and sample standard deviation; use zero std for n<2 and NaN for n=0."""
        n = int(mask.sum().item())
        if n == 0:
            return (float("nan"), float("nan"))
        vals = t[mask]
        mean = float(vals.mean().item())
        std = float(vals.std(unbiased=True).item()) if n >= 2 else 0.0
        return (mean, std)

    results = {}
    for c, (ttype, tier) in enumerate(EVAL_CONDITIONS):
        m = cond == c
        m_has = m & has
        n_env = int(m.sum().item())
        n_valid = int(m_has.sum().item())

        upright = m & (~tipped)
        slide_mask = upright & has  # Only include measured environments whose payload did not tip.
        upright_f = (~tipped).float()  # Per-environment payload-retention indicator.

        row = {"terrain": ttype, "tier": tier, "n_envs": n_env, "n_valid": n_valid}
        for key, (t, mask) in {
            "lin_vel_err_mse": (ev_velmse, m_has),
            "pitch_mse": (ev_pitchmse, m_has),
            "roll_mse": (ev_rollmse, m_has),
            "pitch_abs_max": (max_pitch, m_has),  # Aggregate per-trajectory maxima across environments.
            "roll_abs_max": (max_roll, m_has),
            "abs_vz_mean": (ev_absvz, m_has),
            "abs_vz_max": (max_absvz, m_has),
            "payload_success_rate": (upright_f, m),  # Mean success rate and variation across environments.
            "slide_dist_mean": (slide_max, slide_mask),
        }.items():
            mean, std = _ms(t, mask)
            row[key] = {"mean": mean, "std": std}
        results[f"{ttype}_{tier}"] = row

    _print_tables(results, args_cli.task)

    out_dir = output_path(
        args_cli.output_dir, "logs", "eval", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "num_envs": num_envs,
        "target_speed": args_cli.target_speed,
        "accel_time": args_cli.accel_time,
        "settle_time": args_cli.settle_time,
        "measure_time": args_cli.measure_time,
        "feature_gate_thresh": gate_thr,
        "tip_angle_deg": args_cli.tip_angle_deg,
        "step_dt": step_dt,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2, ensure_ascii=False)
    _write_csv(os.path.join(out_dir, "metrics.csv"), results)
    print(f"\n[INFO] Results saved to: {out_dir}")

    env.close()


# =============================================================================

# =============================================================================
_METRICS = [
    ("lin_vel_err_mse", "lin-vel-err MSE [(m/s)^2]"),
    ("pitch_mse", "pitch MSE [rad^2]"),
    ("roll_mse", "roll MSE [rad^2]"),
    ("pitch_abs_max", "per-env max|pitch| [rad]"),
    ("roll_abs_max", "per-env max|roll| [rad]"),
    ("abs_vz_mean", "mean|v_z| [m/s]"),
    ("abs_vz_max", "per-env max|v_z| [m/s]"),
    ("payload_success_rate", "payload success rate"),
    ("slide_dist_mean", "slide dist (upright) [m]"),
]


def _print_tables(results: dict, task: str):
    print("\n" + "=" * 98)
    print(f"  EVAL RESULTS (mean +/- std over envs)  |  task = {task}")
    print("=" * 98)
    header = "terrain \\ tier".ljust(12) + "".join(f"{t:>21}" for t in EVAL_TIERS)
    for key, label in _METRICS:
        print(f"\n[{label}]")
        print(header)
        for ttype in EVAL_TERRAIN_TYPES:
            row = ttype.ljust(12)
            for tier in EVAL_TIERS:
                d = results[f"{ttype}_{tier}"][key]
                m, s = d["mean"], d["std"]
                cell = f"{m:.4f}+/-{s:.4f}" if m == m else "nan"  # NaN values fail the self-equality check.
                row += f"{cell:>21}"
            print(row)
    print("=" * 98)


def _write_csv(path: str, results: dict):
    cols = ["terrain", "tier", "n_envs", "n_valid"]
    for k, _ in _METRICS:
        cols += [f"{k}_mean", f"{k}_std"]
    lines = [",".join(cols)]
    for ttype in EVAL_TERRAIN_TYPES:
        for tier in EVAL_TIERS:
            r = results[f"{ttype}_{tier}"]
            vals = [r["terrain"], r["tier"], r["n_envs"], r["n_valid"]]
            for k, _ in _METRICS:
                vals += [r[k]["mean"], r[k]["std"]]
            lines.append(",".join(str(v) for v in vals))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
