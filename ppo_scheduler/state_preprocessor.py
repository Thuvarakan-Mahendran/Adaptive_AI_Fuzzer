"""
state_preprocessor.py – Validation, normalization, and coverage helpers.

Responsibilities
----------------
- Define the exact 192-D state slice layout (§4).
- Validate incoming states strictly (§11.1).
- Normalize continuous (non-binary) features with running statistics (§11.2).
- Provide the deterministic edge→bin hash function (§5.3).
- Maintain the rolling coverage window (§5.2).

Not responsible for
-------------------
- Socket I/O
- Serialization / codec
- AFL++ integration
"""

from __future__ import annotations

import math
import struct
from collections import deque
from typing import Dict, Optional

import numpy as np
import torch

from ppo_scheduler.config import PPOConfig

# ---------------------------------------------------------------------------
# Slice layout (§4)
# ---------------------------------------------------------------------------

COVERAGE_SLICE = slice(0, 128)      # 128 binary bits
USAGE_SLICE    = slice(128, 144)    # 16-D operator usage   (§7)
SUCCESS_SLICE  = slice(144, 160)    # 16-D operator success (§8)
LAST_OP_SLICE  = slice(160, 176)    # 16-D last-op one-hot  (§9)
TELEM_SLICE    = slice(176, 192)    # 16-D campaign telem.  (§10)

NUM_OPERATOR_SLOTS = 16
NUM_COVERAGE_BINS  = 128
NUM_TELEM_DIMS     = 16


# ---------------------------------------------------------------------------
# Deterministic edge hash (§5.3)
# ---------------------------------------------------------------------------

def _murmur3_32(key: int) -> int:
    """Pure-Python MurmurHash3 finalizer (32-bit).

    Uses a fixed seed of 0 so results are deterministic across runs.
    Python's built-in hash() is explicitly avoided (§5.3).
    """
    # Pack the integer into 4 bytes (little-endian), then hash
    raw = struct.pack("<I", key & 0xFFFFFFFF)
    h: int = 0  # seed

    c1 = 0xCC9E2D51
    c2 = 0x1B873593

    k = struct.unpack("<I", raw)[0]
    k = (k * c1) & 0xFFFFFFFF
    k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
    k = (k * c2) & 0xFFFFFFFF

    h ^= k
    h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
    h = (h * 5 + 0xE6546B64) & 0xFFFFFFFF

    # finalisation mix
    h ^= 4
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


def edge_to_bin(edge_id: int) -> int:
    """Map an edge identifier to a coverage bin index in [0, 127].

    Uses MurmurHash3 (32-bit) for determinism across processes.
    Hash collisions are expected and harmless (§5.4).

    Args:
        edge_id: Non-negative integer AFL++ edge identifier.

    Returns:
        int in [0, 127].
    """
    return _murmur3_32(edge_id) & (NUM_COVERAGE_BINS - 1)


# ---------------------------------------------------------------------------
# Rolling coverage window (§5.2)
# ---------------------------------------------------------------------------

class CoverageTracker:
    """Maintains a rolling window of recently discovered edges.

    Within the window the 128-bit coverage vector is built from new
    discoveries only (differential / recency-weighted representation).
    At window boundary the tracker resets so stale bits do not persist.

    Args:
        window_size: Number of recent executions to consider (W = 100).
    """

    def __init__(self, window_size: int = 100) -> None:
        self._window_size = window_size
        self._window: deque[set] = deque()  # each entry = set of edges in that exec
        self._current_exec: set[int] = set()

    def record_edge(self, edge_id: int) -> None:
        """Record a newly discovered edge in the current execution."""
        self._current_exec.add(edge_id)

    def commit_execution(self) -> None:
        """Advance the window by one execution step."""
        self._window.append(self._current_exec)
        self._current_exec = set()
        if len(self._window) > self._window_size:
            self._window.popleft()

    def get_coverage_bits(self) -> np.ndarray:
        """Return the 128-D binary coverage vector for the current window.

        A bin is set to 1 if any edge in the rolling window maps to it.

        Returns:
            np.ndarray of shape (128,), dtype float32, values ∈ {0, 1}.
        """
        bits = np.zeros(NUM_COVERAGE_BINS, dtype=np.float32)
        for exec_edges in self._window:
            for edge_id in exec_edges:
                bits[edge_to_bin(edge_id)] = 1.0
        return bits

    def reset(self) -> None:
        """Clear all history (called on env.reset())."""
        self._window.clear()
        self._current_exec = set()


# ---------------------------------------------------------------------------
# Running normalization statistics (§11.2)
# ---------------------------------------------------------------------------

class RunningNormalizer:
    """Online Welford mean/variance estimator for continuous features.

    Applied only to operator usage (16-D), operator success (16-D),
    last-op one-hot (16-D), and telemetry (16-D), i.e. state[128:192].

    Binary coverage (state[0:128]) is NEVER normalized.

    The statistics are saveable/restorable so they can be included in
    checkpoints and restored at evaluation time without data leakage.
    """

    def __init__(self, dim: int, epsilon: float = 1e-8) -> None:
        self._dim = dim
        self._eps = epsilon
        self._n = 0
        self._mean = np.zeros(dim, dtype=np.float64)
        self._M2   = np.ones(dim,  dtype=np.float64)  # init variance = 1

    def update(self, x: np.ndarray) -> None:
        """Update statistics with a new observation.

        Args:
            x: 1-D array of shape (dim,).
        """
        self._n += 1
        delta = x.astype(np.float64) - self._mean
        self._mean += delta / self._n
        delta2 = x.astype(np.float64) - self._mean
        self._M2 += delta * delta2

    @property
    def mean(self) -> np.ndarray:
        return self._mean.astype(np.float32)

    @property
    def std(self) -> np.ndarray:
        if self._n < 2:
            return np.ones(self._dim, dtype=np.float32)
        return np.sqrt(self._M2 / (self._n - 1) + self._eps).astype(np.float32)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Return (x - mean) / std without modifying stored stats."""
        return ((x.astype(np.float32) - self.mean) / (self.std + self._eps))

    def state_dict(self) -> Dict:
        return {
            "n": self._n,
            "mean": self._mean.copy(),
            "M2": self._M2.copy(),
            "dim": self._dim,
        }

    def load_state_dict(self, d: Dict) -> None:
        self._n    = d["n"]
        self._mean = d["mean"].copy()
        self._M2   = d["M2"].copy()
        self._dim  = d["dim"]


# ---------------------------------------------------------------------------
# State validator (§11.1)
# ---------------------------------------------------------------------------

def validate_state(state: np.ndarray, action_dim: int = 16) -> None:
    """Strict validation of a 192-D state vector.

    Raises ValueError or TypeError with clear messages on violation.

    Checks:
    1. dtype-compatible array;
    2. shape is exactly (192,);
    3. all values are finite;
    4. coverage section is binary ({0, 1});
    5. last-operator section is all-zero or valid one-hot;
    6. operator history dimensions are finite;
    7. telemetry dimensions are finite.

    Args:
        state: numpy array of shape (192,).
        action_dim: Number of active mutation operators (≤ 16).
    """
    if not isinstance(state, np.ndarray):
        raise TypeError(f"State must be np.ndarray, got {type(state)}")

    if state.shape != (192,):
        raise ValueError(
            f"State shape must be (192,), got {state.shape}"
        )

    if not np.all(np.isfinite(state)):
        bad_idx = np.where(~np.isfinite(state))[0]
        raise ValueError(
            f"State contains non-finite values at indices {bad_idx.tolist()}"
        )

    # Coverage must be binary
    coverage = state[COVERAGE_SLICE]
    non_binary = np.where(~np.isin(coverage, [0.0, 1.0]))[0]
    if len(non_binary) > 0:
        raise ValueError(
            f"Coverage section (state[0:128]) must be binary {{0,1}}. "
            f"Non-binary values at relative indices {non_binary.tolist()}: "
            f"{coverage[non_binary].tolist()}"
        )

    # Last-operator: all-zero OR exactly one 1.0 in [0:action_dim]
    last_op = state[LAST_OP_SLICE]
    ones = np.where(last_op == 1.0)[0]
    if not (np.all(last_op == 0.0) or
            (len(ones) == 1 and ones[0] < action_dim and
             np.sum(last_op) == 1.0)):
        raise ValueError(
            f"Last-operator section (state[160:176]) must be all-zero or "
            f"a valid one-hot within [0:{action_dim}). Got: {last_op.tolist()}"
        )

    # Operator histories and telemetry: already covered by global finite check
    # but call out specifically for clarity.
    for name, slc in [("usage", USAGE_SLICE),
                      ("success", SUCCESS_SLICE),
                      ("telemetry", TELEM_SLICE)]:
        section = state[slc]
        if not np.all(np.isfinite(section)):
            raise ValueError(
                f"{name} section contains non-finite values: {section.tolist()}"
            )


# ---------------------------------------------------------------------------
# State preprocessor (§11)
# ---------------------------------------------------------------------------

class StatePreprocessor:
    """Validates and normalizes 192-D states before feeding the network.

    Binary coverage (state[0:128]) is never modified.
    Continuous features (state[128:192]) are normalized with running stats.

    Normalization statistics are updated only during training, not evaluation.

    Args:
        config: PPOConfig instance.
    """

    CONTINUOUS_SLICE = slice(128, 192)  # 64-D continuous block
    CONTINUOUS_DIM   = 64

    def __init__(self, config: PPOConfig) -> None:
        self._config = config
        self._normalizer = RunningNormalizer(dim=self.CONTINUOUS_DIM)
        self._training = True

    def train_mode(self) -> None:
        """Enable running-stats updates (training)."""
        self._training = True

    def eval_mode(self) -> None:
        """Freeze normalization statistics (evaluation)."""
        self._training = False

    def validate(self, state: np.ndarray) -> None:
        """Validate a 192-D state.  Raises on violation."""
        validate_state(state, action_dim=self._config.ACTION_DIM)

    def normalize(self, state: np.ndarray) -> np.ndarray:
        """Return a normalized copy of the state.

        Coverage bits are unchanged.  Continuous features are Z-scored.

        If in training mode, updates running statistics before normalizing.

        Args:
            state: Valid 192-D float32 array.

        Returns:
            New 192-D float32 array.
        """
        out = state.copy().astype(np.float32)

        continuous = out[self.CONTINUOUS_SLICE]
        if self._training:
            self._normalizer.update(continuous)
        out[self.CONTINUOUS_SLICE] = self._normalizer.normalize(continuous)
        return out

    def process(self, state: np.ndarray) -> torch.Tensor:
        """Validate, normalize, and convert to a float32 torch.Tensor.

        Args:
            state: Raw 192-D numpy array.

        Returns:
            torch.Tensor of shape (192,), dtype=torch.float32.
        """
        self.validate(state)
        normed = self.normalize(state)
        return torch.from_numpy(normed)

    def state_dict(self) -> Dict:
        """Return serializable normalization statistics."""
        return self._normalizer.state_dict()

    def load_state_dict(self, d: Dict) -> None:
        """Restore normalization statistics from a checkpoint."""
        self._normalizer.load_state_dict(d)
