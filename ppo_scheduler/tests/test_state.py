"""
tests/test_state.py – State validation and slice layout tests (§25.1).
"""

import numpy as np
import pytest

from ppo_scheduler.config import PPOConfig
from ppo_scheduler.state_preprocessor import (
    COVERAGE_SLICE,
    LAST_OP_SLICE,
    SUCCESS_SLICE,
    TELEM_SLICE,
    USAGE_SLICE,
    validate_state,
)


def make_valid_state(action_dim: int = 8) -> np.ndarray:
    """Construct a minimal valid 192-D state."""
    s = np.zeros(192, dtype=np.float32)
    # Binary coverage (some 1s)
    s[0:10] = 1.0
    # L1-normalised usage (sum ≤ 1)
    s[128] = 0.5
    s[129] = 0.5
    # last-op one-hot at index 2
    s[162] = 1.0
    # telemetry: arbitrary finite values in [0,1]
    s[176:192] = 0.5
    return s


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def test_valid_state_passes():
    s = make_valid_state()
    validate_state(s)  # must not raise


def test_wrong_shape_191():
    with pytest.raises(ValueError, match="shape"):
        validate_state(np.zeros(191, dtype=np.float32))


def test_wrong_shape_193():
    with pytest.raises(ValueError, match="shape"):
        validate_state(np.zeros(193, dtype=np.float32))


# ---------------------------------------------------------------------------
# Coverage binary constraint
# ---------------------------------------------------------------------------

def test_non_binary_coverage_raises():
    s = make_valid_state()
    s[5] = 0.5  # not 0 or 1
    with pytest.raises(ValueError, match="binary"):
        validate_state(s)


def test_negative_coverage_raises():
    s = make_valid_state()
    s[10] = -1.0
    with pytest.raises(ValueError, match="binary"):
        validate_state(s)


def test_coverage_value_2_raises():
    s = make_valid_state()
    s[0] = 2.0
    with pytest.raises(ValueError, match="binary"):
        validate_state(s)


# ---------------------------------------------------------------------------
# NaN / Inf
# ---------------------------------------------------------------------------

def test_nan_raises():
    s = make_valid_state()
    s[130] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_state(s)


def test_inf_raises():
    s = make_valid_state()
    s[180] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        validate_state(s)


# ---------------------------------------------------------------------------
# Last-operator section
# ---------------------------------------------------------------------------

def test_all_zero_last_op_valid():
    s = make_valid_state()
    s[LAST_OP_SLICE] = 0.0
    validate_state(s)  # all-zero is valid


def test_valid_one_hot_last_op():
    s = make_valid_state()
    s[LAST_OP_SLICE] = 0.0
    s[160] = 1.0  # operator 0
    validate_state(s)


def test_invalid_last_op_two_ones_raises():
    s = make_valid_state()
    s[LAST_OP_SLICE] = 0.0
    s[160] = 1.0
    s[161] = 1.0  # two ones → invalid
    with pytest.raises(ValueError, match="one-hot"):
        validate_state(s)


# ---------------------------------------------------------------------------
# Slice lengths
# ---------------------------------------------------------------------------

def test_slice_lengths():
    assert len(range(*COVERAGE_SLICE.indices(192))) == 128
    assert len(range(*USAGE_SLICE.indices(192)))    == 16
    assert len(range(*SUCCESS_SLICE.indices(192)))  == 16
    assert len(range(*LAST_OP_SLICE.indices(192)))  == 16
    assert len(range(*TELEM_SLICE.indices(192)))    == 16
