"""
agent.py – PPO agent: action selection, rollout updates, checkpointing (§13, §18, §23).

Public interface
----------------
    agent = PPOAgent(config)
    action, log_prob, value = agent.select_action(state)
    agent.update(rollout_buffer)
    agent.save("checkpoints/step_0100.pt")
    agent.load("checkpoints/step_0100.pt")
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ppo_scheduler.buffer import RolloutBuffer
from ppo_scheduler.config import PPOConfig
from ppo_scheduler.losses import (
    combined_loss,
    entropy_loss,
    ppo_policy_loss,
    value_loss,
)
from ppo_scheduler.networks import Actor, Critic
from ppo_scheduler.state_preprocessor import StatePreprocessor

CHECKPOINT_VERSION = 1


class PPOAgent:
    """PPO mutation-strategy scheduler agent.

    Maintains actor/critic networks with separate optimizers,
    handles action selection and gradient updates, and provides
    checkpoint save/load functionality.

    Args:
        config:       PPOConfig instance.
        preprocessor: StatePreprocessor (shared with the training loop so
                      normalization statistics accumulate correctly).
    """

    def __init__(
        self,
        config: PPOConfig,
        preprocessor: Optional[StatePreprocessor] = None,
    ) -> None:
        self._config = config
        self._device = config.resolve_device()

        self._preprocessor = preprocessor or StatePreprocessor(config)

        # Networks (§12)
        self.actor  = Actor(config).to(self._device)
        self.critic = Critic(config).to(self._device)

        # Separate optimizers with different learning rates (§19)
        self._actor_opt  = torch.optim.Adam(
            self.actor.parameters(), lr=config.ACTOR_LR
        )
        self._critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=config.CRITIC_LR
        )

        self._step = 0  # rollout update counter

    # ------------------------------------------------------------------
    # Action selection (§13)
    # ------------------------------------------------------------------

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """Select a mutation strategy given the current state.

        Steps:
        1. Preprocess (validate + normalize) the state.
        2. Convert to float32 tensor on the correct device.
        3. Pass through actor → Categorical distribution.
        4. Sample action (or argmax in deterministic mode).
        5. Compute log-probability and critic value.

        Args:
            state:         Raw 192-D numpy state array.
            deterministic: If True, select the highest-probability action
                           (used during evaluation, §24).

        Returns:
            (action: int, log_prob: float, value: float)
        """
        processed = self._preprocessor.process(state)          # (192,) float32
        tensor = processed.unsqueeze(0).to(self._device)       # (1, 192)

        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            dist = self.actor.get_distribution(tensor)
            if deterministic:
                action_t = torch.argmax(dist.probs, dim=-1)
            else:
                action_t = dist.sample()
            log_prob_t = dist.log_prob(action_t)
            value_t    = self.critic(tensor)

        action   = int(action_t.item())
        log_prob = float(log_prob_t.item())
        value    = float(value_t.item())

        self.actor.train()
        self.critic.train()
        return action, log_prob, value

    # ------------------------------------------------------------------
    # PPO update (§18)
    # ------------------------------------------------------------------

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Perform PPO_EPOCHS × minibatch updates on the completed rollout.

        1. Bootstrap next value (last stored next_state).
        2. Compute GAE advantages and returns.
        3. Loop PPO_EPOCHS times over shuffled minibatches.
        4. Zero grads → forward → loss → backward → clip → step.
        5. Return mean diagnostic scalars.

        The old_log_probs stored in the buffer are NEVER modified here.

        Args:
            buffer: Completed rollout buffer (is_full() == True).

        Returns:
            Dictionary with mean training statistics for logging.
        """
        cfg = self._config

        # Buffer stores raw (unprocessed) states.
        # During update, use frozen normalizer (eval mode) so running stats
        # don't change while we're iterating over the collected rollout.
        self._preprocessor.eval_mode()

        def _norm_batch(raw_list: list) -> torch.Tensor:
            """Normalize a list of raw 192-D numpy states → float32 tensor."""
            arr = np.stack([
                self._preprocessor.normalize(s) for s in raw_list
            ], axis=0)
            return torch.tensor(arr, dtype=torch.float32).to(self._device)

        # Bootstrap value for the last next_state (normalized, not revalidated)
        last_normed = self._preprocessor.normalize(buffer._next_states[-1])
        last_tensor = torch.tensor(last_normed, dtype=torch.float32
                                   ).unsqueeze(0).to(self._device)
        with torch.no_grad():
            next_value = float(self.critic(last_tensor).item())
        if buffer._dones[-1]:
            next_value = 0.0

        buffer.compute_returns_and_advantages(
            next_value=next_value,
            gamma=cfg.GAMMA,
            gae_lambda=cfg.GAE_LAMBDA,
        )

        # Pre-process all states once before the epoch loop
        all_states_t     = _norm_batch(buffer._states)
        all_actions_t    = torch.tensor(buffer._actions,
                                        dtype=torch.long).to(self._device)
        all_log_probs_t  = torch.tensor(buffer._old_log_probs,
                                        dtype=torch.float32).to(self._device)
        all_advantages_t = buffer._advantages.to(self._device)
        all_returns_t    = buffer._returns.to(self._device)

        stats: Dict[str, list] = {
            "policy_loss": [], "value_loss": [], "entropy": [], "total_loss": [],
        }

        n = len(buffer._states)
        for _ in range(cfg.PPO_EPOCHS):
            indices = np.random.permutation(n)
            for start in range(0, n, cfg.MINIBATCH_SIZE):
                idx = indices[start: start + cfg.MINIBATCH_SIZE]
                if len(idx) == 0:
                    continue

                states_b  = all_states_t[idx]
                actions_b = all_actions_t[idx]
                old_lp_b  = all_log_probs_t[idx]
                adv_b     = all_advantages_t[idx]
                ret_b     = all_returns_t[idx]

                dist_b   = self.actor.get_distribution(states_b)
                new_lp_b = dist_b.log_prob(actions_b)
                ent_b    = entropy_loss(dist_b)
                values_b = self.critic(states_b)

                p_loss = ppo_policy_loss(new_lp_b, old_lp_b, adv_b, cfg.CLIP_EPSILON)
                v_loss = value_loss(values_b, ret_b)
                total  = combined_loss(p_loss, v_loss, ent_b,
                                       cfg.VALUE_COEF, cfg.ENTROPY_COEF)

                if not torch.isfinite(total):
                    raise RuntimeError(f"Non-finite total loss: {total.item()}")

                self._actor_opt.zero_grad()
                self._critic_opt.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    cfg.MAX_GRAD_NORM,
                )
                self._actor_opt.step()
                self._critic_opt.step()

                stats["policy_loss"].append(p_loss.item())
                stats["value_loss"].append(v_loss.item())
                stats["entropy"].append(ent_b.item())
                stats["total_loss"].append(total.item())

        # Restore training mode so future select_action calls update stats
        self._preprocessor.train_mode()
        self._step += 1
        return {k: float(np.mean(v)) for k, v in stats.items()}

    # ------------------------------------------------------------------
    # Checkpointing (§23)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save actor, critic, optimizers, config, and normalizer stats.

        Args:
            path: Absolute or relative file path (e.g. 'checkpoints/step_0010.pt').
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            "version":        CHECKPOINT_VERSION,
            "state_dim":      self._config.STATE_DIM,
            "action_dim":     self._config.ACTION_DIM,
            "step":           self._step,
            "actor_state":    self.actor.state_dict(),
            "critic_state":   self.critic.state_dict(),
            "actor_opt":      self._actor_opt.state_dict(),
            "critic_opt":     self._critic_opt.state_dict(),
            "normalizer":     self._preprocessor.state_dict(),
            "config":         self._config.__dict__.copy(),
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Restore training state from a checkpoint.

        Raises ValueError for incompatible checkpoints (§23).

        Args:
            path: Path to the checkpoint file.
        """
        ckpt = torch.load(path, map_location=self._device)

        if ckpt.get("state_dim") != 192:
            raise ValueError(
                f"Checkpoint STATE_DIM={ckpt.get('state_dim')} ≠ 192."
            )
        if ckpt.get("action_dim") != self._config.ACTION_DIM:
            raise ValueError(
                f"Checkpoint ACTION_DIM={ckpt.get('action_dim')} ≠ "
                f"{self._config.ACTION_DIM}."
            )

        self.actor.load_state_dict(ckpt["actor_state"])
        self.critic.load_state_dict(ckpt["critic_state"])
        self._actor_opt.load_state_dict(ckpt["actor_opt"])
        self._critic_opt.load_state_dict(ckpt["critic_opt"])
        self._preprocessor.load_state_dict(ckpt["normalizer"])
        self._step = ckpt.get("step", 0)
