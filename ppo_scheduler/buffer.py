"""
buffer.py – Rollout buffer with GAE (§14, §16).

Stores one complete rollout of ROLLOUT_LENGTH transitions, then:
  1. computes GAE advantages;
  2. computes returns;
  3. yields shuffled minibatches for PPO updates;
  4. provides a clear() method to reset for the next rollout.

The old_log_probs stored at collection time are NEVER replaced during
the PPO update phase (§18).
"""

from __future__ import annotations

from typing import Generator, Tuple

import numpy as np
import torch

from ppo_scheduler.config import PPOConfig


class RolloutBuffer:
    """Fixed-length rollout buffer for on-policy PPO.

    Args:
        config: PPOConfig instance.
        device:  torch device string used for tensor creation.
    """

    def __init__(self, config: PPOConfig, device: str = "cpu") -> None:
        self._config = config
        self._device = device
        self._capacity = config.ROLLOUT_LENGTH

        # Pre-allocated storage (filled sequentially)
        self._states:        list[np.ndarray] = []
        self._actions:       list[int]        = []
        self._rewards:       list[float]      = []
        self._dones:         list[bool]       = []
        self._values:        list[float]      = []
        self._old_log_probs: list[float]      = []
        self._next_states:   list[np.ndarray] = []

        # Computed by compute_returns_and_advantages()
        self._advantages: torch.Tensor | None = None
        self._returns:    torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Collection API
    # ------------------------------------------------------------------

    def add(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        done:       bool,
        value:      float,
        log_prob:   float,
        next_state: np.ndarray,
    ) -> None:
        """Store a single transition.

        Args:
            state:      Pre-processed 192-D numpy state at time t.
            action:     Selected action index.
            reward:     Observed scalar reward.
            done:       Whether the episode terminated.
            value:      Critic estimate V(s_t) at collection time.
            log_prob:   Log-probability of the selected action (old policy).
            next_state: Pre-processed 192-D numpy state at time t+1.
        """
        if len(self._states) >= self._capacity:
            raise RuntimeError(
                "RolloutBuffer is full.  Call clear() before adding more."
            )
        self._states.append(state.copy())
        self._actions.append(int(action))
        self._rewards.append(float(reward))
        self._dones.append(bool(done))
        self._values.append(float(value))
        self._old_log_probs.append(float(log_prob))
        self._next_states.append(next_state.copy())

    def is_full(self) -> bool:
        """True when the buffer holds exactly ROLLOUT_LENGTH transitions."""
        return len(self._states) == self._capacity

    # ------------------------------------------------------------------
    # GAE + returns (§16)
    # ------------------------------------------------------------------

    def compute_returns_and_advantages(
        self,
        next_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute GAE advantages and TD-λ returns in-place.

        δ_t = r_t + γ V(s_{t+1})(1-d_t) - V(s_t)
        A_t = δ_t + γλ(1-d_t) A_{t+1}
        R_t = A_t + V(s_t)

        Advantages are normalized after computation (mean=0, std=1).

        Args:
            next_value: Bootstrap value V(s_T) for the last state in the
                        rollout (zero if terminal, critic value otherwise).
            gamma:      Discount factor γ.
            gae_lambda: GAE λ parameter.
        """
        n = len(self._states)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0

        values = np.array(self._values, dtype=np.float32)
        rewards = np.array(self._rewards, dtype=np.float32)
        dones = np.array(self._dones, dtype=np.float32)

        for t in reversed(range(n)):
            if t == n - 1:
                next_val = next_value
                not_done = 1.0 - float(dones[t])
            else:
                next_val = values[t + 1]
                not_done = 1.0 - float(dones[t])

            delta = rewards[t] + gamma * next_val * not_done - values[t]
            gae = delta + gamma * gae_lambda * not_done * gae
            advantages[t] = gae

        returns = advantages + values

        # Normalize advantages (§16)
        adv_mean = advantages.mean()
        adv_std  = advantages.std()
        eps = 1e-8
        advantages = (advantages - adv_mean) / (adv_std + eps)

        self._advantages = torch.tensor(advantages, dtype=torch.float32)
        self._returns    = torch.tensor(returns,    dtype=torch.float32)

    # ------------------------------------------------------------------
    # Minibatch sampling (§18)
    # ------------------------------------------------------------------

    def get_minibatches(
        self, batch_size: int
    ) -> Generator[
        Tuple[
            torch.Tensor,  # states
            torch.Tensor,  # actions
            torch.Tensor,  # old_log_probs
            torch.Tensor,  # advantages
            torch.Tensor,  # returns
        ],
        None,
        None,
    ]:
        """Yield randomly shuffled minibatches of the stored rollout.

        compute_returns_and_advantages() must be called first.

        Args:
            batch_size: Number of transitions per minibatch.

        Yields:
            (states, actions, old_log_probs, advantages, returns)
        """
        if self._advantages is None or self._returns is None:
            raise RuntimeError(
                "Call compute_returns_and_advantages() before get_minibatches()."
            )

        n = len(self._states)
        indices = np.random.permutation(n)

        states_t = torch.tensor(
            np.array(self._states), dtype=torch.float32
        ).to(self._device)
        actions_t = torch.tensor(
            self._actions, dtype=torch.long
        ).to(self._device)
        log_probs_t = torch.tensor(
            self._old_log_probs, dtype=torch.float32
        ).to(self._device)
        advantages_t = self._advantages.to(self._device)
        returns_t    = self._returns.to(self._device)

        for start in range(0, n, batch_size):
            idx = indices[start: start + batch_size]
            if len(idx) == 0:
                continue
            yield (
                states_t[idx],
                actions_t[idx],
                log_probs_t[idx],
                advantages_t[idx],
                returns_t[idx],
            )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all stored transitions and computed quantities."""
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._dones.clear()
        self._values.clear()
        self._old_log_probs.clear()
        self._next_states.clear()
        self._advantages = None
        self._returns    = None

    def __len__(self) -> int:
        return len(self._states)
