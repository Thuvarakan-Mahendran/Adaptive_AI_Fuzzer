"""
test_reward.py – Unit tests for reward.py (Step 6).

Tests:
  - compute_dl_reward: correct formula, edge cases, boundary values
  - compute_dl_reward_from_snapshot: snapshot dict wrapper
  - Reward formula matches aggregator.c: same coefficients
  - Reward is always non-negative for valid inputs
"""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dl_seed_optimizer.reward import (
    LAMBDA_CRASH,
    LAMBDA_NOVELTY,
    compute_dl_reward,
    compute_dl_reward_from_snapshot,
)


class TestComputeDLReward:
    """Tests for compute_dl_reward()."""

    def test_zero_reward_nothing_new(self):
        """No new edges, no crash → reward should be 0.0."""
        r = compute_dl_reward(
            prev_total_edges=100,
            curr_total_edges=100,
            prev_crashes=0,
            curr_crashes=0,
            total_execs=100,
        )
        assert r == pytest.approx(0.0, abs=1e-9)

    def test_crash_reward_only(self):
        """New crash, no new edges → reward = LAMBDA_CRASH = 10.0."""
        r = compute_dl_reward(
            prev_total_edges=100,
            curr_total_edges=100,
            prev_crashes=0,
            curr_crashes=1,
            total_execs=100,
        )
        assert r == pytest.approx(LAMBDA_CRASH, rel=1e-6)

    def test_novelty_reward_only(self):
        """New edges (no crash) → reward = norm_cov + LAMBDA_NOVELTY."""
        delta = 5
        execs = 500
        r = compute_dl_reward(
            prev_total_edges=100,
            curr_total_edges=100 + delta,
            prev_crashes=0,
            curr_crashes=0,
            total_execs=execs,
        )
        expected_norm_cov = delta / math.log(math.e + execs)
        expected = expected_norm_cov + LAMBDA_NOVELTY
        assert r == pytest.approx(expected, rel=1e-6)

    def test_crash_and_novelty(self):
        """Both crash and novelty → both contributions summed."""
        delta = 3
        execs = 1000
        r = compute_dl_reward(
            prev_total_edges=50,
            curr_total_edges=50 + delta,
            prev_crashes=0,
            curr_crashes=1,
            total_execs=execs,
        )
        norm_cov = delta / math.log(math.e + execs)
        expected = norm_cov + LAMBDA_CRASH + LAMBDA_NOVELTY
        assert r == pytest.approx(expected, rel=1e-6)

    def test_reward_always_non_negative(self):
        """Reward is always >= 0 for valid inputs."""
        for execs in [1, 10, 100, 10_000]:
            r = compute_dl_reward(0, 0, 0, 0, execs)
            assert r >= 0.0

    def test_reward_decreases_with_more_execs(self):
        """Same delta_cov → smaller norm_cov reward as total_execs grows."""
        r_early = compute_dl_reward(0, 10, 0, 0, 10)
        r_late = compute_dl_reward(0, 10, 0, 0, 100_000)
        assert r_early > r_late

    def test_larger_delta_cov_gives_larger_reward(self):
        """More new edges → larger reward (all else equal)."""
        r_small = compute_dl_reward(0, 5, 0, 0, 100)
        r_large = compute_dl_reward(0, 50, 0, 0, 100)
        assert r_large > r_small

    def test_negative_delta_cov_clamped_to_zero(self):
        """If curr < prev edges (shouldn't happen but must not crash), delta is clamped."""
        # With clamping: delta_cov = max(0, -10) = 0
        r = compute_dl_reward(
            prev_total_edges=100,
            curr_total_edges=90,  # curr < prev (edge case)
            prev_crashes=0,
            curr_crashes=0,
            total_execs=100,
        )
        assert r == pytest.approx(0.0, abs=1e-9)

    def test_lambda_crash_value(self):
        """LAMBDA_CRASH must be 10.0 to match aggregator.c."""
        assert LAMBDA_CRASH == pytest.approx(10.0)

    def test_lambda_novelty_value(self):
        """LAMBDA_NOVELTY must be 1.0 to match aggregator.c."""
        assert LAMBDA_NOVELTY == pytest.approx(1.0)

    def test_invalid_total_execs_negative(self):
        """Negative total_execs raises ValueError."""
        with pytest.raises(ValueError):
            compute_dl_reward(0, 0, 0, 0, -1)

    def test_zero_execs(self):
        """total_execs=0 is valid (denominator: log(e + 0) = 1)."""
        r = compute_dl_reward(0, 5, 0, 0, 0)
        assert r > 0.0  # Should be finite and positive

    def test_log_normalization_denominator_always_gte_one(self):
        """log(e + total_execs) >= 1 for all total_execs >= 0.
        At total_execs=0: log(e + 0) = log(e) = 1.0 exactly."""
        for execs in [0, 1, 100, 1_000_000]:
            denom = math.log(math.e + execs)
            assert denom >= 1.0

    def test_matches_aggregator_formula_known_values(self):
        """Verify against hand-computed aggregator.c formula result.

        aggregator.c formula:
          norm_cov = delta_cov / log(e + total_execs)
          R = norm_cov + LAMBDA_CRASH*i_crash + LAMBDA_NOVELTY*i_novel

        Test case: delta_cov=10, total_execs=100, crash=True, novel=True
          denom = log(e + 100) = log(102.718...) ≈ 4.632
          norm_cov = 10 / 4.632 ≈ 2.159
          R = 2.159 + 10*1 + 1*1 = 13.159
        """
        delta = 10
        execs = 100
        expected_denom = math.log(math.e + execs)
        expected_norm_cov = delta / expected_denom
        expected_r = expected_norm_cov + 10.0 + 1.0

        r = compute_dl_reward(0, delta, 0, 1, execs)
        assert r == pytest.approx(expected_r, rel=1e-6)


class TestComputeDLRewardFromSnapshot:
    """Tests for compute_dl_reward_from_snapshot()."""

    def test_basic_snapshot_wrapper(self):
        """Snapshot dict wrapper produces same result as direct call."""
        prev = {"total_edges": 100, "crashes": 0}
        curr = {"total_edges": 110, "crashes": 1}
        direct = compute_dl_reward(100, 110, 0, 1, 200)
        via_snap = compute_dl_reward_from_snapshot(prev, curr, total_execs=200)
        assert via_snap == pytest.approx(direct, rel=1e-9)

    def test_no_change(self):
        """No changes → reward = 0."""
        snap = {"total_edges": 50, "crashes": 0}
        r = compute_dl_reward_from_snapshot(snap, snap, total_execs=10)
        assert r == pytest.approx(0.0, abs=1e-9)

    def test_type_coercion(self):
        """Integer-like values in snapshot dicts are coerced correctly."""
        prev = {"total_edges": 0, "crashes": 0}
        curr = {"total_edges": 5, "crashes": 0}
        r = compute_dl_reward_from_snapshot(prev, curr, total_execs=1)
        assert r > 0.0
