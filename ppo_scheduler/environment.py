"""
environment.py – Deterministic mock mutation environment (§20, §21).

This module is the stand-in for the future AFL++ feedback path.
It does NOT implement:
  - actual AFL++ execution
  - Unix sockets
  - state serialization

The environment:
  - maintains operator usage/success statistics
  - simulates coverage progress (rolling-window based)
  - simulates crash and novelty events
  - computes the reward formula R_t = Δcoverage + λ_crash * I_crash + λ_novelty * I_novel
  - exposes all reward components in the info dict for testing

State layout (192 dimensions)
------------------------------
  [0:128]   binary coverage  (via CoverageTracker)
  [128:144] operator usage   (EMA, L1-normalized)
  [144:160] operator success (EMA)
  [160:176] last-op one-hot
  [176:192] campaign telemetry (16 scalars, normalized)
"""

from __future__ import annotations

import random
from typing import Dict, Optional, Tuple

import numpy as np

from ppo_scheduler.config import PPOConfig
from ppo_scheduler.state_preprocessor import (
    CoverageTracker,
    NUM_OPERATOR_SLOTS,
)

# ---------------------------------------------------------------------------
# Telemetry slot indices (§10)
# ---------------------------------------------------------------------------
TELEM_QUEUE_DEPTH           = 0
TELEM_EXECS_SINCE_COV       = 1
TELEM_EXEC_RATE             = 2
TELEM_SEED_ENERGY           = 3
TELEM_ELAPSED_FRACTION      = 4
TELEM_CRASH_COUNT           = 5
TELEM_COV_GROWTH            = 6
TELEM_RECENT_REWARD         = 7
TELEM_TIMEOUT_RATE          = 8
TELEM_MUTATION_EFFICIENCY   = 9
TELEM_QUEUE_GROWTH          = 10
TELEM_TIME_SINCE_CRASH      = 11
TELEM_UNIQUE_CRASH_RATE     = 12
TELEM_AVG_EXEC_TIME         = 13
TELEM_SEED_AGE              = 14
TELEM_RESERVED              = 15

# ---------------------------------------------------------------------------
# Per-operator simulated properties
# Operators with lower index tend to discover coverage faster; index 3 and 7
# have elevated crash probability to allow tests to verify crash bonuses.
# ---------------------------------------------------------------------------
_COVERAGE_PROB  = [0.8, 0.5, 0.6, 0.7, 0.3, 0.9, 0.4, 0.6,
                   0.2, 0.5, 0.7, 0.4, 0.8, 0.3, 0.6, 0.5]
_CRASH_PROB     = [0.02, 0.01, 0.01, 0.05, 0.01, 0.01, 0.01, 0.04,
                   0.01, 0.02, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01]
_NOVELTY_PROB   = [0.3,  0.2,  0.25, 0.15, 0.1,  0.35, 0.2,  0.25,
                   0.1,  0.2,  0.3,  0.15, 0.3,  0.1,  0.2,  0.2]


class MockMutationEnvironment:
    """Deterministic/testable stand-in for the AFL++ feedback path.

    Args:
        config:        PPOConfig instance.
        max_steps:     Episode length (steps before done=True).
        rng_seed:      Seed for reproducibility.
    """

    USAGE_DECAY   = 0.99   # EMA decay for operator usage   (§7)
    SUCCESS_DECAY = 0.99   # EMA decay for operator success (§8)
    # ε for L1 normalization
    L1_EPS        = 1e-8
    # Simulated edge-ID range
    NUM_EDGES     = 4096

    def __init__(
        self,
        config:    PPOConfig,
        max_steps: int = 500,
        rng_seed:  int = 0,
    ) -> None:
        self._config    = config
        self._max_steps = max_steps
        self._rng       = random.Random(rng_seed)
        self._np_rng    = np.random.default_rng(rng_seed)
        self._cov_tracker = CoverageTracker(window_size=config.COVERAGE_WINDOW)

        # Internal mutable state
        self._usage:   np.ndarray = np.zeros(NUM_OPERATOR_SLOTS, dtype=np.float32)
        self._success: np.ndarray = np.zeros(NUM_OPERATOR_SLOTS, dtype=np.float32)
        self._last_op: int = -1   # -1 = no action yet

        self._step_count:           int   = 0
        self._total_coverage_bits:  int   = 0
        self._prev_coverage_bits:   int   = 0
        self._crash_count:          int   = 0
        self._novelty_count:        int   = 0
        self._execs_since_coverage: int   = 0
        self._recent_reward:        float = 0.0
        self._queue_depth:          int   = 10

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial 192-D state.

        Returns:
            np.ndarray of shape (192,), dtype float32.
        """
        self._cov_tracker.reset()
        self._usage   = np.zeros(NUM_OPERATOR_SLOTS, dtype=np.float32)
        self._success = np.zeros(NUM_OPERATOR_SLOTS, dtype=np.float32)
        self._last_op = -1

        self._step_count           = 0
        self._total_coverage_bits  = 0
        self._prev_coverage_bits   = 0
        self._crash_count          = 0
        self._novelty_count        = 0
        self._execs_since_coverage = 0
        self._recent_reward        = 0.0
        self._queue_depth          = 10

        return self._build_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one mutation step.

        Args:
            action: Integer in [0, ACTION_DIM).

        Returns:
            (next_state, reward, done, info)
            info exposes Δcoverage, I_crash, I_novelty for testing.
        """
        if action < 0 or action >= self._config.ACTION_DIM:
            raise ValueError(
                f"action={action} out of range [0, {self._config.ACTION_DIM})"
            )

        cfg = self._config

        # ---- Simulate operator outcomes -----------------------------------
        op_idx = action  # operator index matches action
        cov_p     = _COVERAGE_PROB[op_idx]
        crash_p   = _CRASH_PROB[op_idx]
        novelty_p = _NOVELTY_PROB[op_idx]

        # Simulate newly discovered edges this execution
        num_new_edges = int(self._np_rng.binomial(20, cov_p))
        edges_discovered = [
            int(self._np_rng.integers(0, self.NUM_EDGES))
            for _ in range(num_new_edges)
        ]
        for e in edges_discovered:
            self._cov_tracker.record_edge(e)
        self._cov_tracker.commit_execution()

        # Coverage change
        cov_bits_now    = int(self._cov_tracker.get_coverage_bits().sum())
        delta_coverage  = float(cov_bits_now - self._prev_coverage_bits)
        self._prev_coverage_bits = cov_bits_now
        self._total_coverage_bits = cov_bits_now

        if num_new_edges > 0:
            self._execs_since_coverage = 0
        else:
            self._execs_since_coverage += 1

        # Crash event
        i_crash = 1 if self._np_rng.random() < crash_p else 0
        if i_crash:
            self._crash_count += 1

        # Novelty event
        i_novelty = 1 if self._np_rng.random() < novelty_p else 0
        if i_novelty:
            self._novelty_count += 1

        # ---- Reward (§15) -------------------------------------------------
        reward = (
            delta_coverage
            + cfg.LAMBDA_CRASH   * i_crash
            + cfg.LAMBDA_NOVELTY * i_novelty
        )
        self._recent_reward = reward

        # ---- Update operator statistics -----------------------------------
        # Usage EMA (§7): u_i ← 0.99*u_i + I[a_t=i]
        self._usage *= self.USAGE_DECAY
        self._usage[op_idx] += 1.0

        # Success EMA (§8): s_i ← decay*s_i + (1-decay)*signal
        success_signal = 1.0 if (i_crash == 1 or i_novelty == 1 or delta_coverage > 0) else 0.0
        self._success[op_idx] = (
            self.SUCCESS_DECAY * self._success[op_idx]
            + (1.0 - self.SUCCESS_DECAY) * success_signal
        )

        # Last operator
        self._last_op = op_idx

        # Queue depth simulation
        if delta_coverage > 0:
            self._queue_depth = min(self._queue_depth + int(delta_coverage), 1000)

        self._step_count += 1
        done = (self._step_count >= self._max_steps)

        next_state = self._build_state()
        info = {
            "delta_coverage":  delta_coverage,
            "i_crash":         i_crash,
            "i_novelty":       i_novelty,
            "reward":          reward,
            "step":            self._step_count,
            "coverage_bits":   cov_bits_now,
        }
        return next_state, reward, done, info

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def _build_state(self) -> np.ndarray:
        """Construct the full 192-D state vector."""
        state = np.zeros(192, dtype=np.float32)

        # [0:128] binary coverage (§5)
        state[0:128] = self._cov_tracker.get_coverage_bits()

        # [128:144] operator usage, L1-normalized (§7)
        usage_norm = self._usage / (self._usage.sum() + self.L1_EPS)
        state[128:144] = usage_norm

        # [144:160] operator success EMA (§8)
        state[144:160] = self._success

        # [160:176] last-op one-hot (§9)
        if self._last_op >= 0:
            state[160 + self._last_op] = 1.0

        # [176:192] campaign telemetry (§10)
        state[176:192] = self._build_telemetry()

        return state

    def _build_telemetry(self) -> np.ndarray:
        """Return 16 normalized campaign telemetry scalars (§10).

        Each value is clipped or normalized to a bounded range before
        entering the neural network.  Unavailable metrics use a
        deterministic placeholder (0.0) and are documented below.
        """
        t = np.zeros(16, dtype=np.float32)
        cfg = self._config

        # 0: queue depth → normalized to [0, 1] over max 1000
        t[TELEM_QUEUE_DEPTH]         = min(self._queue_depth / 1000.0, 1.0)
        # 1: execs since last new coverage → clip at 500, normalize
        t[TELEM_EXECS_SINCE_COV]     = min(self._execs_since_coverage / 500.0, 1.0)
        # 2: normalized execution rate → placeholder 0.5 (no real timer)
        t[TELEM_EXEC_RATE]           = 0.5
        # 3: seed energy → placeholder 0.5
        t[TELEM_SEED_ENERGY]         = 0.5
        # 4: elapsed campaign fraction → step / max_steps
        t[TELEM_ELAPSED_FRACTION]    = self._step_count / max(self._max_steps, 1)
        # 5: recent crash count → normalize over 10 crashes max
        t[TELEM_CRASH_COUNT]         = min(self._crash_count / 10.0, 1.0)
        # 6: recent coverage growth → coverage bits / 128
        t[TELEM_COV_GROWTH]          = self._total_coverage_bits / 128.0
        # 7: recent reward (clipped to [-1, 1] after scaling)
        t[TELEM_RECENT_REWARD]       = np.clip(self._recent_reward / 20.0, -1.0, 1.0)
        # 8: timeout rate → placeholder 0.0 (no real timeout tracking)
        t[TELEM_TIMEOUT_RATE]        = 0.0
        # 9: mutation efficiency → success mean over active operators
        active = self._success[: cfg.ACTION_DIM]
        t[TELEM_MUTATION_EFFICIENCY] = float(active.mean())
        # 10: queue growth → clipped
        t[TELEM_QUEUE_GROWTH]        = min(self._queue_depth / 500.0, 1.0)
        # 11: time since last crash → placeholder 0.5
        t[TELEM_TIME_SINCE_CRASH]    = 0.5
        # 12: unique crash rate → crash_count / step_count
        t[TELEM_UNIQUE_CRASH_RATE]   = self._crash_count / max(self._step_count, 1)
        # 13: average execution time → placeholder 0.3
        t[TELEM_AVG_EXEC_TIME]       = 0.3
        # 14: seed age → placeholder 0.5
        t[TELEM_SEED_AGE]            = 0.5
        # 15: reserved → 0.0
        t[TELEM_RESERVED]            = 0.0

        return t
