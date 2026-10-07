"""
losses.py – PPO clipped policy loss, value loss, and entropy (§17).

All PPO math lives in this module so it is easy to test independently
and there is no duplication between the agent and the training loop.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch.distributions import Categorical


def ppo_policy_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages:    torch.Tensor,
    clip_epsilon:  float = 0.2,
) -> torch.Tensor:
    """Clipped PPO policy (surrogate) loss.

    L^CLIP = -E[ min(r_t * A_t,  clip(r_t, 1-ε, 1+ε) * A_t) ]

    where r_t = exp(new_log_prob - old_log_prob).

    Args:
        new_log_probs: Log-probs from the *current* policy,  shape (N,).
        old_log_probs: Log-probs from the *collection* policy, shape (N,).
                       These must NOT be updated during the PPO epochs.
        advantages:    Normalized advantage estimates, shape (N,).
        clip_epsilon:  ε parameter (default 0.2, §17).

    Returns:
        Scalar loss tensor (minimized by the optimizer → gradient ascent
        on the objective is handled by the negation inside).
    """
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    return policy_loss


def value_loss(
    predicted_values: torch.Tensor,
    returns:           torch.Tensor,
) -> torch.Tensor:
    """Mean-squared-error critic loss.

    L^V = MSE(V(s), R_t)

    Args:
        predicted_values: Critic output, shape (N,) or (N, 1).
        returns:           Target returns, shape (N,).

    Returns:
        Scalar loss tensor.
    """
    return F.mse_loss(predicted_values.squeeze(-1), returns)


def entropy_loss(distribution: Categorical) -> torch.Tensor:
    """Mean entropy of the categorical action distribution.

    Used as a bonus to encourage exploration (subtracted from total loss).

    Args:
        distribution: torch.distributions.Categorical.

    Returns:
        Scalar mean entropy tensor (positive; caller subtracts it).
    """
    return distribution.entropy().mean()


def combined_loss(
    policy_l:    torch.Tensor,
    value_l:     torch.Tensor,
    entropy_l:   torch.Tensor,
    value_coef:  float = 0.5,
    entropy_coef: float = 0.01,
) -> torch.Tensor:
    """Combined PPO objective.

    total_loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

    Args:
        policy_l:     Clipped policy loss (positive, to minimize).
        value_l:      Value MSE loss (positive, to minimize).
        entropy_l:    Mean entropy (positive; subtracted to maximize).
        value_coef:   Coefficient on value loss (0.5, §17).
        entropy_coef: Coefficient on entropy bonus (0.01, §17).

    Returns:
        Scalar combined loss tensor.
    """
    return policy_l + value_coef * value_l - entropy_coef * entropy_l
