"""
networks.py – Actor and Critic feed-forward networks (§12).

Architecture
------------
Actor:  192 → 256 → 256 → ACTION_DIM  (outputs logits for Categorical)
Critic: 192 → 256 → 256 → 1           (outputs scalar V(s))

Restrictions (§12, §31)
-----------------------
- No RNN/LSTM/GRU
- No attention
- No DQN / Q-values
- No UCB
- No separate policy heads
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from ppo_scheduler.config import PPOConfig


class Actor(nn.Module):
    """Policy network: state → action logits.

    Architecture: Linear(192→256) → Tanh → Linear(256→256) → Tanh →
                  Linear(256→ACTION_DIM)

    Tanh is used over ReLU so that activations stay bounded, which tends
    to improve PPO stability on small state dimensions.

    Args:
        config: PPOConfig instance.
    """

    def __init__(self, config: PPOConfig) -> None:
        super().__init__()
        self._action_dim = config.ACTION_DIM
        self.net = nn.Sequential(
            nn.Linear(config.STATE_DIM, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, config.ACTION_DIM),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # Smaller gain on the output layer → initial policy close to uniform
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return action logits.

        Args:
            x: Tensor of shape (batch, 192) or (192,).

        Returns:
            Tensor of shape (batch, ACTION_DIM) or (ACTION_DIM,).
        """
        return self.net(x)

    def get_distribution(self, x: torch.Tensor) -> Categorical:
        """Build a Categorical distribution from state tensor.

        Args:
            x: Tensor of shape (batch, 192).

        Returns:
            torch.distributions.Categorical over ACTION_DIM actions.
        """
        logits = self.forward(x)
        return Categorical(logits=logits)


class Critic(nn.Module):
    """Value network: state → scalar V(s).

    Architecture: Linear(192→256) → Tanh → Linear(256→256) → Tanh →
                  Linear(256→1)

    Args:
        config: PPOConfig instance.
    """

    def __init__(self, config: PPOConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.STATE_DIM, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return scalar value estimate V(s).

        Args:
            x: Tensor of shape (batch, 192) or (192,).

        Returns:
            Tensor of shape (batch, 1) or (1,).
        """
        return self.net(x)
