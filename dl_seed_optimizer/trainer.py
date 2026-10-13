"""
trainer.py – DQN training logic for the DL Seed Optimizer.

Implements DQN with experience replay and a target network.

Training algorithm:
    1. Sample batch of 32 transitions from ReplayBuffer
    2. Compute Q_current = online_net(tokens, lengths)          [32]
    3. Compute TD target:
         Q_next   = target_net(next_tokens, next_lengths)       [32]  (no_grad)
         Q_target = rewards + γ * Q_next * (1 - dones)          [32]
    4. loss = Huber(Q_current, Q_target)
    5. optimizer.zero_grad() → loss.backward() → clip_grad_norm(1.0) → step()
    6. Every 1,000 gradient steps: hard-copy online → target (model_version++)

Hyperparameters (from DLConfig):
    LEARNING_RATE = 0.001        Paper-specified (spec §27)
    GAMMA         = 0.95         Paper-specified (spec §23)
    BUFFER_CAPACITY = 50,000     Implementation choice (spec §20)
    BATCH_SIZE    = 32           Implementation choice (spec §21)
    TARGET_SYNC   = 1,000        Implementation choice (spec §28)
    MAX_GRAD_NORM = 1.0          Implementation choice
    MIN_BUFFER_SIZE = 1,000      Implementation choice
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

from dl_seed_optimizer.config import DLConfig
from dl_seed_optimizer.models import SeedQNetwork
from dl_seed_optimizer.replay_buffer import ReplayBuffer, collate_transitions

logger = logging.getLogger("dl_seed_optimizer.trainer")


class DQNTrainer:
    """Handles DQN training: online network, target network, replay buffer.

    Instantiate once; call train_step() from the background training thread.
    The online network is accessed by the inference (IPC handler) thread for
    Q-value scoring — PyTorch CPU forward passes are thread-safe for reading.

    Attributes:
        online_net:      The online SeedQNetwork (updated every gradient step).
        target_net:      The target SeedQNetwork (synced every TARGET_SYNC steps).
        buffer:          ReplayBuffer for experience storage.
        gradient_steps:  Total gradient updates performed.
        model_version:   Increments on each target network sync.
    """

    def __init__(self, config: DLConfig) -> None:
        """Initialize DQNTrainer.

        Args:
            config: DLConfig with all hyperparameters.
        """
        self.config = config
        device_str = config.resolve_device()
        self.device = torch.device(device_str)

        # Online network: trained every gradient step
        self.online_net = SeedQNetwork(config).to(self.device)

        # Target network: hard-copied from online every TARGET_SYNC steps
        # Never trained directly; always in eval mode.
        self.target_net = SeedQNetwork(config).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Adam optimizer — lr=0.001 is paper-specified
        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(),
            lr=config.LEARNING_RATE,
        )

        # Huber loss (SmoothL1Loss): more robust to outliers than MSE
        self.loss_fn = nn.SmoothL1Loss()

        # Replay buffer — 50,000 capacity circular buffer
        self.buffer = ReplayBuffer(capacity=config.BUFFER_CAPACITY)

        # Counters
        self.gradient_steps: int = 0
        self.model_version: int = 0

        # ε-greedy state for exploration
        self.epsilon: float = config.EPSILON_START

        logger.info(
            "DQNTrainer initialized on %s | lr=%.4f | γ=%.3f | "
            "buffer_cap=%d | batch=%d | target_sync=%d",
            device_str,
            config.LEARNING_RATE,
            config.GAMMA,
            config.BUFFER_CAPACITY,
            config.BATCH_SIZE,
            config.TARGET_SYNC,
        )

    def train_step(self) -> Optional[Dict[str, float]]:
        """Perform one DQN gradient update from the replay buffer.

        Returns None if buffer is not yet large enough to train.
        Returns a metrics dict otherwise.

        This method is called by the background training thread.
        It is safe to call concurrently with online_net.forward() reads
        from the main thread (PyTorch CPU tensor writes are non-corrupting).

        Returns:
            None if buffer has fewer than MIN_BUFFER_SIZE transitions.
            Dict with keys: 'loss', 'q_mean', 'q_std', 'gradient_steps',
                            'model_version', 'buffer_size', 'epsilon'.
        """
        if not self.buffer.is_ready(self.config.MIN_BUFFER_SIZE):
            return None

        # Sample batch
        batch = self.buffer.sample(self.config.BATCH_SIZE)
        data = collate_transitions(batch)

        # Move to device
        tokens = data["tokens"].to(self.device)
        lengths = data["lengths"].to(self.device)
        rewards = data["rewards"].to(self.device)
        dones = data["dones"].to(self.device)
        next_tokens = data["next_tokens"].to(self.device)
        next_lengths = data["next_lengths"].to(self.device)

        # ── Forward pass (online network) ────────────────────────────────
        self.online_net.train()
        q_current = self.online_net(tokens, lengths)  # [B]

        # ── TD target (target network, no gradient) ───────────────────────
        with torch.no_grad():
            q_next = self.target_net(next_tokens, next_lengths)  # [B]
            # Q_target = r + γ * Q_next * (1 - done)
            # If done=1 (episode boundary), future Q is masked to 0.
            q_target = rewards + self.config.GAMMA * q_next * (1.0 - dones)

        # ── Huber loss ────────────────────────────────────────────────────
        loss = self.loss_fn(q_current, q_target)

        # ── Backpropagation ───────────────────────────────────────────────
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(),
            max_norm=self.config.MAX_GRAD_NORM,
        )
        self.optimizer.step()

        self.gradient_steps += 1

        # ── Target network sync ───────────────────────────────────────────
        if self.gradient_steps % self.config.TARGET_SYNC == 0:
            self._sync_target_network()

        # ── ε-greedy decay ────────────────────────────────────────────────
        self.epsilon = max(
            self.config.EPSILON_MIN,
            self.epsilon * self.config.EPSILON_DECAY,
        )

        # ── Metrics ───────────────────────────────────────────────────────
        if self.gradient_steps % self.config.LOG_INTERVAL == 0:
            logger.info(
                "step=%d | loss=%.6f | q_mean=%.4f | model_ver=%d | "
                "buffer=%d | ε=%.4f",
                self.gradient_steps,
                loss.item(),
                q_current.mean().item(),
                self.model_version,
                len(self.buffer),
                self.epsilon,
            )

        return {
            "loss": loss.item(),
            "q_mean": q_current.mean().item(),
            "q_std": q_current.std().item(),
            "gradient_steps": float(self.gradient_steps),
            "model_version": float(self.model_version),
            "buffer_size": float(len(self.buffer)),
            "epsilon": self.epsilon,
        }

    def _sync_target_network(self) -> None:
        """Hard-copy online network weights to target network.

        Increments model_version so the Q-value cache in dl_daemon.py
        knows to discard stale cached Q-values.
        """
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.model_version += 1
        logger.info(
            "Target network synced at gradient_step=%d (model_version=%d)",
            self.gradient_steps,
            self.model_version,
        )

    def save_checkpoint(self, path: str) -> None:
        """Save online network weights and trainer state to a checkpoint file.

        Args:
            path: File path for the checkpoint (.pt file).
        """
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # Note: DLConfig is not stored in the checkpoint to keep it
        # compatible with PyTorch 2.6+ weights_only=True loading.
        # Callers must pass the same DLConfig when loading.
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "gradient_steps": self.gradient_steps,
                "model_version": self.model_version,
                "epsilon": self.epsilon,
            },
            path,
        )
        logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: str) -> None:
        """Load weights and trainer state from a checkpoint file.

        Args:
            path: Path to the checkpoint file to load.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
        """
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # weights_only=False: safe here since we only load our own checkpoints.
        # PyTorch 2.6+ changed the default to weights_only=True, which
        # breaks loading of checkpoints containing non-tensor scalars.
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.target_net.eval()
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.gradient_steps = ckpt.get("gradient_steps", 0)
        self.model_version = ckpt.get("model_version", 0)
        self.epsilon = ckpt.get("epsilon", self.config.EPSILON_START)
        logger.info(
            "Checkpoint loaded from %s (step=%d, model_ver=%d)",
            path, self.gradient_steps, self.model_version,
        )
