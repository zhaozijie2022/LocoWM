"""Action-conditioned prediction of future task-state chunks.

WM(obs_t, action_t) predicts [s_(t+1), ..., s_(t+H)] with shape
[B, chunk_len, num_targets]. The wm_target observation contains angular
velocity, world-frame linear acceleration and projected gravity (nine values)."""

from __future__ import annotations

import torch
import torch.nn as nn

from loco_rl.utils import resolve_nn_activation


class _ResidualBlock(nn.Module):
    """Constant-width residual block: act(x + W2 act(W1 x))."""

    def __init__(self, dim: int, activation: nn.Module):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.act = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.lin1(x))
        h = self.lin2(h)
        return self.act(x + h)


class WorldModel(nn.Module):
    """Train task-state predictions alongside PPO using its rollout buffer.

    PPO initializes the model with build(), records each environment step, and
    calls update() before clearing the shared rollout storage."""

    def __init__(self, cfg: dict):
        super().__init__()

        self.chunk_len = int(cfg.get("chunk_len", 5))
        self.num_targets = int(cfg.get("num_targets", 9))
        # Current-state reconstruction uses a single target instead of a future chunk.
        self.reconstruct_current = bool(cfg.get("reconstruct_current", False))
        if self.reconstruct_current:
            self.chunk_len = 1
        self.hidden_dims = list(cfg.get("mlp_hidden_dims", [512, 256, 128]))
        self.activation_name = cfg.get("activation", "elu")
        self.use_resnet = bool(cfg.get("use_resnet", False))
        self.learning_rate = float(cfg.get("learning_rate", 1.0e-3))
        self.num_learning_epochs = int(cfg.get("num_learning_epochs", 5))
        self.num_mini_batches = int(cfg.get("num_mini_batches", 4))
        self.max_grad_norm = cfg.get("max_grad_norm", 1.0)

        # Initialized by build().
        self.net: nn.Sequential | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.targets: torch.Tensor | None = None  # [T, N, num_targets]
        self.step = 0
        self.device = "cpu"
        self._num_obs: int | None = None
        self._num_actions: int | None = None

        self.last_loss = 0.0

    # region build
    def build(
        self,
        num_obs: int,
        num_actions: int,
        num_envs: int | None = None,
        num_transitions_per_env: int | None = None,
        device="cpu",
        for_training: bool = True,
    ):
        """Build the network and, for training, Adam and a [T, N, targets] buffer.

        Stage 2 sets for_training=False to construct only the frozen predictor."""
        self.device = device
        self._num_obs = num_obs
        self._num_actions = num_actions
        input_dim = num_obs + num_actions
        output_dim = self.chunk_len * self.num_targets
        self.net = self._build_mlp(input_dim, self.hidden_dims, output_dim, self.activation_name, self.use_resnet)
        self.net.to(device)
        if for_training:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
            # Keep target writes synchronized with the rollout storage.
            self.targets = torch.zeros(num_transitions_per_env, num_envs, self.num_targets, device=device)
        self.step = 0
        print(f"World Model MLP (in={input_dim}, out={output_dim} = {self.chunk_len}x{self.num_targets}): {self.net}")

    @staticmethod
    def _build_mlp(input_dim, hidden_dims, output_dim, activation_name, use_resnet):
        activation = resolve_nn_activation(activation_name)
        if not use_resnet:
            layers = [nn.Linear(input_dim, hidden_dims[0]), activation]
            for i in range(len(hidden_dims)):
                if i == len(hidden_dims) - 1:
                    layers.append(nn.Linear(hidden_dims[i], output_dim))
                else:
                    layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                    layers.append(activation)
            return nn.Sequential(*layers)
        # Input projection, constant-width residual blocks, then output head.
        width = hidden_dims[0]
        modules: list[nn.Module] = [nn.Linear(input_dim, width), activation]
        for _ in range(len(hidden_dims)):
            modules.append(_ResidualBlock(width, resolve_nn_activation(activation_name)))
        modules.append(nn.Linear(width, output_dim))
        return nn.Sequential(*modules)

    # region record/forward
    def record_step(self, infos: dict):
        """Record the post-step wm_target observation, corresponding to state t+1."""
        target = infos["observations"]["wm_target"].to(self.device)
        self.targets[self.step].copy_(target)
        self.step += 1

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """[B, obs] , [B, act] -> [B, chunk_len, num_targets]。"""
        x = torch.cat([obs, actions], dim=-1)
        out = self.net(x)
        return out.view(out.shape[0], self.chunk_len, self.num_targets)

    # region update
    def _build_chunks(self, dones: torch.Tensor):
        """Build future targets and masks with shapes [T, N, H, targets] and [T, N, H, 1].

        Offset k uses targets[t+k], valid only when done[t:t+k+1] contains no reset.
        Offsets beyond the rollout are masked out."""
        T, N, _ = self.targets.shape
        H = self.chunk_len
        notdone = 1.0 - dones.float()  # [T, N, 1]
        chunk_tgt = torch.zeros(T, N, H, self.num_targets, device=self.device)
        chunk_mask = torch.zeros(T, N, H, 1, device=self.device)
        cum = torch.ones(T, N, 1, device=self.device)
        for k in range(H):
            if k == 0:
                shifted_tgt = self.targets
                nd_k = notdone
            else:
                shifted_tgt = torch.zeros_like(self.targets)
                shifted_tgt[: T - k] = self.targets[k:]
                nd_k = torch.zeros_like(notdone)  # Mask offsets beyond the rollout.
                nd_k[: T - k] = notdone[k:]
            cum = cum * nd_k  # A reset invalidates this and all later targets in the chunk.
            chunk_tgt[:, :, k] = shifted_tgt
            chunk_mask[:, :, k] = cum
        return chunk_tgt, chunk_mask

    def _build_current_targets(self):
        """Build current-state targets for reconstruction (H=1).

        Since targets[t] stores state t+1, use targets[t-1] for observation t.
        Mask t=0 because no previous target is available. Return targets of shape
        [T, N, 1, num_targets] and masks of shape [T, N, 1, 1]."""
        T, N, _ = self.targets.shape
        cur_tgt = torch.zeros(T, N, 1, self.num_targets, device=self.device)
        cur_mask = torch.zeros(T, N, 1, 1, device=self.device)
        cur_tgt[1:, :, 0] = self.targets[:-1]  # cur_tgt[t] = targets[t-1] = s_t
        cur_mask[1:] = 1.0  # No previous state is available for t=0.
        return cur_tgt, cur_mask

    def update(self, storage) -> float:
        """Update predictions from the shared PPO rollout and recorded targets.

        Reset the model target-buffer counter without clearing PPO storage."""
        T, N = self.targets.shape[0], self.targets.shape[1]
        obs = storage.observations  # [T, N, num_obs]
        actions = storage.actions  # [T, N, num_actions]
        dones = storage.dones  # [T, N, 1] (byte)
        if self.reconstruct_current:
            chunk_tgt, chunk_mask = self._build_current_targets()
        else:
            chunk_tgt, chunk_mask = self._build_chunks(dones)

        # Flatten time and environment dimensions.
        obs_f = obs.reshape(T * N, -1)
        act_f = actions.reshape(T * N, -1)
        tgt_f = chunk_tgt.reshape(T * N, self.chunk_len, self.num_targets)
        mask_f = chunk_mask.reshape(T * N, self.chunk_len, 1)

        batch_size = T * N
        mini_batch_size = batch_size // self.num_mini_batches
        total_loss = 0.0
        count = 0
        for _ in range(self.num_learning_epochs):
            indices = torch.randperm(self.num_mini_batches * mini_batch_size, device=self.device)
            for i in range(self.num_mini_batches):
                idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
                o, a = obs_f[idx], act_f[idx]
                t, m = tgt_f[idx], mask_f[idx]

                pred = self.forward(o, a)  # [B, H, num_targets]
                sq_err = (pred - t) ** 2 * m
                denom = m.sum() * self.num_targets + 1e-8  # Number of valid scalar targets.
                loss = sq_err.sum() / denom

                if not torch.isfinite(loss):
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                count += 1

        self.last_loss = total_loss / max(count, 1)
        self.step = 0  # Keep the counter synchronized with storage.clear().
        return self.last_loss
