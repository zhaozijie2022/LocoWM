<div align="center">

### LocoWM: High-Precision Locomotion through World-Model-Guided Residual Adaptation

**CASIA**

<p align="center">
  <strong><a href="https://zhaozijie2022.github.io/locowm/">Project Website</a></strong> · <strong><a href="https://arxiv.org/pdf/2609.39179">Paper PDF</a></strong> · <strong><a href="https://arxiv.org/abs/2609.39179">arXiv</a></strong>
</p>

<p align="right">
  🌎 <a href="README.md">English</a> | <a href="README_CN.md">中文</a>
</p>

</div>

## Overview

![LocoWM: transporting unsecured payloads across terrain, during acceleration, and under external pushes](docs/assets/images/teaser.png)

LocoWM helps robots maintain precise control while moving. A world model predicts the effects of a base policy's proposed action, and a residual adapter uses these predictions to correct anticipated deviations before execution.

## Installation

Environment: Python 3.11, Isaac Sim 5.1.0, Isaac Lab commit `c91a125c73`, PyTorch 2.7.0 / CUDA 12.8, and RSL-RL 2.3.3.

**1. Install the simulation environment.** Run from this repository's root; Isaac Lab is cloned into a sibling directory. If you already have a working `isaaclab` environment, skip to step 2.

```bash
conda create -n isaaclab python=3.11
conda activate isaaclab

python -m pip install "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/isaac-sim/IsaacLab.git ../IsaacLab
pushd ../IsaacLab
git checkout c91a125c73c8b574878419a9583afc0b63b99f0a
./isaaclab.sh --install none
popd
```

**2. Install LocoWM.**

```bash
conda activate isaaclab
python -m pip install -e .
python -m locowm.scripts.list_envs
```

The commands below use TensorBoard. For W&B or Neptune, run `python -m pip install -e '.[logging]'` and select `--logger wandb` or `--logger neptune`.

## Train a LocoWM policy

**Stage 1: base policy and world model.**

```bash
python -m locowm.scripts.train \
  --task Isaac-LocomotionGo2W-v1 \
  --num_envs 4096 --headless \
  --logger tensorboard --run_name stage1
```

Checkpoints are saved in pairs under `logs/rsl_rl/locomotion/<run>/`. Replace the placeholders with your checkpoint details:

```bash
STAGE1_RUN="<stage1-run>"
STAGE1_ITER="<iteration>"
POLICY_CHECKPOINT="logs/rsl_rl/locomotion/${STAGE1_RUN}/model_${STAGE1_ITER}.pt"
WORLD_MODEL_CHECKPOINT="logs/rsl_rl/locomotion/${STAGE1_RUN}/world_model_${STAGE1_ITER}.pt"
```

**Stage 2: residual adapter.** Freeze the Stage 1 policy and world model, then train the adapter.

```bash
python -m locowm.scripts.train \
  --task Isaac-TransportGo2W-Adapter-v1 \
  --num_envs 4096 --headless \
  --logger tensorboard --run_name stage2 \
  --policy_checkpoint "$POLICY_CHECKPOINT" \
  --world_model_checkpoint "$WORLD_MODEL_CHECKPOINT"
```

Checkpoints are saved under `logs/rsl_rl/transport_adapter/<run>/`. Playback and evaluation reuse the Stage 1 files and shell variables above.

## Playback and evaluation

Select the Stage 2 run directory name and checkpoint filename:

```bash
STAGE2_RUN="<stage2-run>"
STAGE2_CHECKPOINT="model_<iteration>.pt"
```

**Policy playback.**

```bash
python -m locowm.scripts.play \
  --task Isaac-TransportGo2W-Adapter-Play-v1 --num_envs 20 \
  --load_run "$STAGE2_RUN" --checkpoint "$STAGE2_CHECKPOINT" \
  --policy_checkpoint "$POLICY_CHECKPOINT" \
  --world_model_checkpoint "$WORLD_MODEL_CHECKPOINT"
```

To inspect the base policy, use `Isaac-LocomotionGo2W-Play-v1` with a Stage 1 run/checkpoint and omit `--policy_checkpoint` and `--world_model_checkpoint`.

**Control metrics.** Measure velocity tracking, posture, and payload behavior across four terrain types and four difficulty levels.

```bash
python -m locowm.scripts.eval \
  --task Isaac-EvalGo2W-Adapter-v1 --num_envs 256 --headless \
  --load_run "$STAGE2_RUN" --checkpoint "$STAGE2_CHECKPOINT" \
  --policy_checkpoint "$POLICY_CHECKPOINT" \
  --world_model_checkpoint "$WORLD_MODEL_CHECKPOINT"
```

Results are saved as `metrics.json` and `metrics.csv` under `logs/eval/<task>/<timestamp>/`.

**Payload success rate.**

```bash
python -m locowm.scripts.succ_eval \
  --task Isaac-SuccGo2W-Adapter-v1 --num_envs 256 --headless \
  --load_run "$STAGE2_RUN" --checkpoint "$STAGE2_CHECKPOINT" \
  --policy_checkpoint "$POLICY_CHECKPOINT" \
  --world_model_checkpoint "$WORLD_MODEL_CHECKPOINT"
```

A trial succeeds when the robot crosses the requested distance without dropping the payload. Drops during initial acceleration are retried and excluded from the success/failure counts. Results are saved as `success.json` and `success.csv` under `logs/succ_eval/<task>/<timestamp>/`.

## Alternative variants

**End-to-end transport policy.**

```bash
python -m locowm.scripts.train \
  --task Isaac-TransportGo2W-v1 \
  --num_envs 4096 --headless --logger tensorboard
```

**Reactive residual.** Use the Stage 1 base policy with zero world-model features; omit `--world_model_checkpoint`.

```bash
python -m locowm.scripts.train \
  --task Isaac-TransportGo2W-Adapter-NoWM-v1 \
  --num_envs 4096 --headless --logger tensorboard \
  --policy_checkpoint "$POLICY_CHECKPOINT"
```

**Reconstruction.** Replace future prediction with current-state reconstruction. First train the reconstruction variant of Stage 1:

```bash
python -m locowm.scripts.train \
  --task Isaac-LocomotionGo2W-ReconWM-v1 \
  --num_envs 4096 --headless --logger tensorboard
```

Then train the adapter using the policy and world model from the same iteration of that run:

```bash
python -m locowm.scripts.train \
  --task Isaac-TransportGo2W-Adapter-ReconWM-v1 \
  --num_envs 4096 --headless --logger tensorboard \
  --policy_checkpoint "logs/rsl_rl/locomotion_reconwm/<run>/model_<iteration>.pt" \
  --world_model_checkpoint "logs/rsl_rl/locomotion_reconwm/<run>/world_model_<iteration>.pt"
```

## Code and research notes

`locowm` contains the Isaac Lab tasks and entry points. `loco_rl` contains the world model, residual adapter, and training extensions; shared components come directly from RSL-RL.

This project builds on [Isaac Lab](https://github.com/isaac-sim/IsaacLab), [RSL-RL](https://github.com/leggedrobotics/rsl_rl), [robot_lab](https://github.com/fan-ziqi/robot_lab), and [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym). We thank their authors and contributors.
