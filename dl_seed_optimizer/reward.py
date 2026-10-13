"""
reward.py – Reward computation for the DL Seed Optimizer.

Reward formula (matches adaptive-fuzzer/c_mutator/aggregator.c):

    R = norm_cov + LAMBDA_CRASH * I_crash + LAMBDA_NOVELTY * I_novel

Where:
    norm_cov  = delta_coverage_bits / log(e + total_execs)
    I_crash   = 1 if new crash found, else 0
    I_novel   = 1 if delta_coverage > 0, else 0

Weights:
    LAMBDA_CRASH   = 10.0  (matches aggregator.c line 22)
    LAMBDA_NOVELTY =  1.0  (matches aggregator.c line 23)

This module computes rewards in Python independently of the C aggregator.
It uses identical coefficients for cross-module consistency.
"""

from __future__ import annotations

import math
from typing import Optional

# Reward coefficients — MUST match aggregator.c and ppo_scheduler/config.py
LAMBDA_CRASH: float = 10.0
LAMBDA_NOVELTY: float = 1.0


def compute_dl_reward(
    prev_total_edges: int,
    curr_total_edges: int,
    prev_crashes: int,
    curr_crashes: int,
    total_execs: int,
) -> float:
    """Compute seed-level reward after one fuzzing round.

    This is the DL module's per-seed reward, analogous to the per-execution
    reward in aggregator.c's agg_compute_reward(). Same formula, same
    coefficients, computed in Python.

    Args:
        prev_total_edges: Number of unique AFL++ bitmap edges set before
                          fuzzing this seed.
        curr_total_edges: Number of unique edges set after fuzzing.
        prev_crashes:     Total crash count before fuzzing this seed.
        curr_crashes:     Total crash count after fuzzing this seed.
        total_execs:      Total campaign executions (monotonically increasing).
                          Used in log-normalization to guard against reward
                          growth as coverage saturates.

    Returns:
        Scalar reward R ∈ [0, +∞). Zero if no new coverage or crashes.

    Examples:
        >>> compute_dl_reward(100, 105, 0, 0, 500)   # 5 new edges, no crash
        5 / log(e + 500) + 0 + 1.0
        >>> compute_dl_reward(100, 100, 0, 1, 500)   # crash found, no new edges
        0.0 + 10.0 + 0.0 = 10.0
        >>> compute_dl_reward(100, 100, 0, 0, 500)   # nothing new
        0.0
    """
    if total_execs < 0:
        raise ValueError(f"total_execs must be >= 0, got {total_execs}")

    delta_cov = max(0, curr_total_edges - prev_total_edges)

    # Log-normalize to guard against reward growth as coverage saturates.
    # Denominator is always > 1 since math.e > 1.
    norm_cov: float = delta_cov / math.log(math.e + total_execs)

    i_crash: float = 1.0 if curr_crashes > prev_crashes else 0.0
    i_novel: float = 1.0 if delta_cov > 0 else 0.0

    return norm_cov + LAMBDA_CRASH * i_crash + LAMBDA_NOVELTY * i_novel


def compute_dl_reward_from_snapshot(
    prev_snapshot: dict,
    curr_snapshot: dict,
    total_execs: int,
) -> float:
    """Convenience wrapper accepting snapshot dicts.

    Snapshot dict keys: 'total_edges' (int), 'crashes' (int).

    Args:
        prev_snapshot: State before fuzzing (keys: 'total_edges', 'crashes').
        curr_snapshot: State after fuzzing (keys: 'total_edges', 'crashes').
        total_execs:   Total campaign execution count.

    Returns:
        Scalar reward float.
    """
    return compute_dl_reward(
        prev_total_edges=int(prev_snapshot["total_edges"]),
        curr_total_edges=int(curr_snapshot["total_edges"]),
        prev_crashes=int(prev_snapshot["crashes"]),
        curr_crashes=int(curr_snapshot["crashes"]),
        total_execs=int(total_execs),
    )


def normalize_reward(reward: float, running_mean: float, running_std: float,
                     epsilon: float = 1e-8) -> float:
    """Optionally normalize a reward using running statistics.

    Not used by default but provided for experimentation.

    Args:
        reward:       Raw reward value.
        running_mean: Exponential moving average of reward.
        running_std:  Exponential moving standard deviation.
        epsilon:      Numerical stability term.

    Returns:
        Normalized reward: (reward - mean) / (std + epsilon).
    """
    return (reward - running_mean) / (running_std + epsilon)
