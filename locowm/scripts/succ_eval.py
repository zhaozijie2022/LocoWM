"""Evaluate payload retention over repeated single-feature traversal trials.

A trial succeeds after the traversal distance with the payload retained.
Payload drops after acceleration and timeouts count as failures. Drops
during acceleration trigger retries and do not enter the success/failure
counts. Report per-environment mean/std and pooled counts for each terrain
and difficulty condition. See README for example commands."""

import argparse
import os
import sys

if __package__ in {None, ""}:  # Support the legacy ``python locowm/scripts/succ_eval.py`` form.
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from isaaclab.app import AppLauncher
from locowm.scripts import cli_args
from locowm.scripts.runtime import experiment_log_root, output_path

parser = argparse.ArgumentParser(description="Payload success-rate eval for an RSL-RL policy.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments (best a multiple of 16).")
parser.add_argument("--task", type=str, default=None, help="Name of the success-eval task.")
parser.add_argument(
    "--target_speed", type=float, default=1.0, help="Forward velocity after the acceleration ramp (m/s)."
)
parser.add_argument("--accel_time", type=float, default=2.5, help="Time to ramp from zero to target_speed (s).")
parser.add_argument(
    "--traverse_distance",
    type=float,
    default=5.0,
    help="Forward displacement required for success with the payload retained (m).",
)
parser.add_argument(
    "--run_steps", type=int, default=5000, help="Total simulation steps; longer runs allow more trials per environment."
)
parser.add_argument(
    "--output_dir", type=str, default=None, help="Output directory (default: logs/succ_eval/<task>/<time>)."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import locowm

locowm.ensure_runtime()


import gymnasium as gym
import json
from datetime import datetime

import torch

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from locowm.config.go2w.eval_env_cfg import EVAL_CONDITIONS, EVAL_TERRAIN_TYPES, EVAL_TIERS


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    step_dt = env_cfg.decimation * env_cfg.sim.dt
    accel_steps = max(1, round(args_cli.accel_time / step_dt))
    env_cfg.commands.base_velocity.target_speed = float(args_cli.target_speed)
    env_cfg.commands.base_velocity.accel_steps = int(accel_steps)
    env_cfg.terminations.traversed_success.params["distance"] = float(args_cli.traverse_distance)

    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = experiment_log_root(agent_cfg.experiment_name)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    from loco_rl.runners import OnPolicyRunner

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    cond = env.unwrapped.scene.terrain.terrain_types.clone().to(device).long()
    tm = env.unwrapped.termination_manager

    print(
        f"[INFO] num_envs={num_envs}, accel_steps={accel_steps}, traverse_distance={args_cli.traverse_distance}, "
        f"run_steps={args_cli.run_steps}",
        flush=True,
    )

    z = lambda: torch.zeros(num_envs, device=device)
    n_succ = z()
    n_dropfail = z()
    n_timeout = z()
    n_acceldrop = z()  # Retries only; excluded from success/failure counts.

    obs, _ = env.get_observations()
    for step in range(args_cli.run_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            dones = dones.bool()
            if dones.any():
                succ = tm.get_term("traversed_success") & dones
                dropfail = tm.get_term("payload_dropped_fail") & dones
                timeout = tm.get_term("time_out") & dones
                acceldrop = tm.get_term("payload_dropped_accel") & dones
                n_succ += succ.float()
                n_dropfail += dropfail.float()
                n_timeout += timeout.float()
                n_acceldrop += acceldrop.float()

    # Per-environment success rates.
    n_trials = n_succ + n_dropfail + n_timeout  # Exclude payload drops during acceleration from completed trials.
    has = n_trials > 0
    ev_rate = torch.where(has, n_succ / n_trials.clamp(min=1.0), torch.full_like(n_succ, float("nan")))

    # Aggregate by terrain condition.
    def _ms(t, mask):
        n = int(mask.sum().item())
        if n == 0:
            return (float("nan"), float("nan"))
        vals = t[mask]
        return (float(vals.mean().item()), float(vals.std(unbiased=True).item()) if n >= 2 else 0.0)

    results = {}
    for c, (ttype, tier) in enumerate(EVAL_CONDITIONS):
        m = cond == c
        m_has = m & has
        tot_trials = float(n_trials[m].sum().item())
        tot_succ = float(n_succ[m].sum().item())
        rate_mean, rate_std = _ms(ev_rate, m_has)
        results[f"{ttype}_{tier}"] = {
            "terrain": ttype,
            "tier": tier,
            "n_envs": int(m.sum().item()),
            "n_trials": int(tot_trials),
            "n_success": int(tot_succ),
            "n_dropfail": int(n_dropfail[m].sum().item()),
            "n_timeout": int(n_timeout[m].sum().item()),
            "n_acceldrop_uncounted": int(n_acceldrop[m].sum().item()),
            "success_rate_pooled": (tot_succ / tot_trials) if tot_trials > 0 else float("nan"),
            "success_rate_mean": rate_mean,  # Mean per-environment success rate.
            "success_rate_std": rate_std,  # Standard deviation of per-environment success rates.
        }

    _print_tables(results, args_cli.task)

    out_dir = output_path(
        args_cli.output_dir, "logs", "succ_eval", args_cli.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "num_envs": num_envs,
        "target_speed": args_cli.target_speed,
        "accel_time": args_cli.accel_time,
        "traverse_distance": args_cli.traverse_distance,
        "run_steps": args_cli.run_steps,
        "step_dt": step_dt,
    }
    with open(os.path.join(out_dir, "success.json"), "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2, ensure_ascii=False)
    _write_csv(os.path.join(out_dir, "success.csv"), results)
    print(f"\n[INFO] Results saved to: {out_dir}")

    env.close()


_CSV_COLS = [
    "terrain",
    "tier",
    "n_envs",
    "n_trials",
    "n_success",
    "n_dropfail",
    "n_timeout",
    "n_acceldrop_uncounted",
    "success_rate_pooled",
    "success_rate_mean",
    "success_rate_std",
]


def _print_tables(results: dict, task: str):
    print("\n" + "=" * 98)
    print(f"  PAYLOAD SUCCESS RATE  |  task = {task}")
    print("=" * 98)
    header = "terrain \\ tier".ljust(12) + "".join(f"{t:>21}" for t in EVAL_TIERS)

    print("\n[success rate  (per-env mean +/- std)]")
    print(header)
    for ttype in EVAL_TERRAIN_TYPES:
        row = ttype.ljust(12)
        for tier in EVAL_TIERS:
            r = results[f"{ttype}_{tier}"]
            m, s = r["success_rate_mean"], r["success_rate_std"]
            cell = f"{m:.3f}+/-{s:.3f}" if m == m else "nan"
            row += f"{cell:>21}"
        print(row)

    print("\n[success rate  (pooled)   /   n_success / n_trials]")
    print(header)
    for ttype in EVAL_TERRAIN_TYPES:
        row = ttype.ljust(12)
        for tier in EVAL_TIERS:
            r = results[f"{ttype}_{tier}"]
            p = r["success_rate_pooled"]
            cell = f"{p:.3f} ({r['n_success']}/{r['n_trials']})" if p == p else "nan"
            row += f"{cell:>21}"
        print(row)
    print("=" * 98)


def _write_csv(path: str, results: dict):
    lines = [",".join(_CSV_COLS)]
    for ttype in EVAL_TERRAIN_TYPES:
        for tier in EVAL_TIERS:
            r = results[f"{ttype}_{tier}"]
            lines.append(",".join(str(r[c]) for c in _CSV_COLS))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
