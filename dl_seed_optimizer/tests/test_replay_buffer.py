"""
test_replay_buffer.py – Unit tests for replay_buffer.py (Step 4).

Tests:
  - ReplayBuffer: capacity, FIFO eviction, circular wrapping
  - ReplayBuffer: thread-safe concurrent push/sample
  - ReplayBuffer: sampling statistics (uniform-ish)
  - collate_transitions: correct padding, shapes, dtypes
  - Transition: dataclass construction
"""

from __future__ import annotations

import os
import sys
import threading

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dl_seed_optimizer.replay_buffer import (
    ReplayBuffer,
    Transition,
    collate_transitions,
)
from dl_seed_optimizer.tokenizer import tokenize_fixed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_transition(
    seed_id: int = 0,
    trace_len: int = 5,
    next_trace_len: int = 4,
    reward: float = 1.0,
    done: bool = False,
) -> Transition:
    """Create a synthetic Transition for testing."""
    tokens = tokenize_fixed(
        np.random.randint(0, 65535, size=trace_len, dtype=np.uint16)
    )
    next_tokens = tokenize_fixed(
        np.random.randint(0, 65535, size=next_trace_len, dtype=np.uint16)
    )
    return Transition(
        seed_id=seed_id,
        trace_tokens=tokens,
        trace_len=trace_len,
        reward=reward,
        next_trace_tokens=next_tokens,
        next_trace_len=next_trace_len,
        done=done,
    )


# ---------------------------------------------------------------------------
# ReplayBuffer tests
# ---------------------------------------------------------------------------

class TestReplayBufferCapacity:
    """Tests for buffer capacity and FIFO eviction."""

    def test_grows_up_to_capacity(self):
        """Buffer accepts transitions up to capacity without eviction."""
        buf = ReplayBuffer(capacity=10)
        for i in range(10):
            buf.push(make_transition(seed_id=i))
        assert len(buf) == 10

    def test_does_not_exceed_capacity(self):
        """Buffer length never exceeds capacity."""
        buf = ReplayBuffer(capacity=5)
        for i in range(20):
            buf.push(make_transition(seed_id=i))
        assert len(buf) == 5

    def test_circular_overwrite(self):
        """Oldest transitions are overwritten when capacity is exceeded."""
        cap = 3
        buf = ReplayBuffer(capacity=cap)
        for i in range(5):
            buf.push(make_transition(seed_id=i, reward=float(i)))
        # Only the 3 most recent (i=2,3,4) should remain
        sample = buf.sample(cap)
        rewards = {t.reward for t in sample}
        assert rewards == {2.0, 3.0, 4.0}

    def test_total_pushed_increments(self):
        """total_pushed counter increases with every push."""
        buf = ReplayBuffer(capacity=10)
        for i in range(7):
            buf.push(make_transition())
        assert buf.total_pushed == 7

    def test_clear_resets_state(self):
        """clear() empties buffer and resets counters."""
        buf = ReplayBuffer(capacity=10)
        for _ in range(5):
            buf.push(make_transition())
        buf.clear()
        assert len(buf) == 0
        assert buf.total_pushed == 0

    def test_invalid_capacity_raises(self):
        """capacity < 1 raises ValueError."""
        with pytest.raises(ValueError):
            ReplayBuffer(capacity=0)


class TestReplayBufferSampling:
    """Tests for sampling behavior."""

    def test_sample_returns_correct_count(self):
        """Sample returns exactly batch_size transitions."""
        buf = ReplayBuffer(capacity=100)
        for i in range(50):
            buf.push(make_transition(seed_id=i))
        batch = buf.sample(batch_size=16)
        assert len(batch) == 16

    def test_sample_capped_at_buffer_size(self):
        """If batch_size > buffer size, returns all stored transitions."""
        buf = ReplayBuffer(capacity=100)
        for _ in range(5):
            buf.push(make_transition())
        batch = buf.sample(batch_size=100)
        assert len(batch) == 5

    def test_sample_empty_buffer_raises(self):
        """Sampling from empty buffer raises ValueError."""
        buf = ReplayBuffer(capacity=10)
        with pytest.raises(ValueError):
            buf.sample(batch_size=1)

    def test_is_ready_false_when_underfilled(self):
        """is_ready() is False when buffer has fewer than min_size entries."""
        buf = ReplayBuffer(capacity=100)
        for _ in range(5):
            buf.push(make_transition())
        assert not buf.is_ready(10)

    def test_is_ready_true_when_sufficient(self):
        """is_ready() is True when buffer has at least min_size entries."""
        buf = ReplayBuffer(capacity=100)
        for _ in range(10):
            buf.push(make_transition())
        assert buf.is_ready(10)


class TestReplayBufferThreadSafety:
    """Tests for concurrent push/sample from multiple threads."""

    def test_concurrent_push_no_corruption(self):
        """Multiple threads pushing simultaneously don't corrupt the buffer."""
        buf = ReplayBuffer(capacity=1000)
        errors = []

        def push_many():
            try:
                for _ in range(100):
                    buf.push(make_transition())
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=push_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(buf) <= 1000  # Must not exceed capacity
        assert buf.total_pushed == 500

    def test_concurrent_push_and_sample(self):
        """Push and sample simultaneously without deadlock or data corruption."""
        buf = ReplayBuffer(capacity=200)

        # Pre-fill buffer so sampling is possible
        for _ in range(50):
            buf.push(make_transition())

        errors = []
        samples_taken = []

        def push_work():
            for _ in range(50):
                buf.push(make_transition())

        def sample_work():
            for _ in range(10):
                try:
                    s = buf.sample(batch_size=8)
                    samples_taken.append(len(s))
                except Exception as e:
                    errors.append(str(e))

        t1 = threading.Thread(target=push_work)
        t2 = threading.Thread(target=sample_work)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0


# ---------------------------------------------------------------------------
# collate_transitions tests
# ---------------------------------------------------------------------------

class TestCollateTransitions:
    """Tests for collate_transitions()."""

    def _make_batch(self, trace_lengths, next_lengths=None):
        """Helper to create a batch with specified trace lengths."""
        if next_lengths is None:
            next_lengths = trace_lengths
        return [
            make_transition(
                seed_id=i,
                trace_len=tl,
                next_trace_len=nl,
                reward=float(i),
                done=(i % 5 == 0),
            )
            for i, (tl, nl) in enumerate(zip(trace_lengths, next_lengths))
        ]

    def test_tokens_shape(self):
        """'tokens' tensor has shape [B, max_trace_len]."""
        batch = self._make_batch([5, 3, 7])
        data = collate_transitions(batch)
        assert data["tokens"].shape == (3, 7)  # max is 7

    def test_next_tokens_shape(self):
        """'next_tokens' tensor has shape [B, max_next_trace_len]."""
        batch = self._make_batch([5, 3, 7], [4, 6, 2])
        data = collate_transitions(batch)
        assert data["next_tokens"].shape == (3, 6)  # max next is 6

    def test_lengths_correct(self):
        """'lengths' tensor matches input trace lengths."""
        batch = self._make_batch([5, 3, 7])
        data = collate_transitions(batch)
        assert data["lengths"].tolist() == [5, 3, 7]

    def test_rewards_correct(self):
        """'rewards' tensor has correct float values."""
        batch = self._make_batch([3, 3, 3])
        data = collate_transitions(batch)
        assert data["rewards"].tolist() == pytest.approx([0.0, 1.0, 2.0])

    def test_dones_float32(self):
        """'dones' tensor is float32 with values 0.0 or 1.0."""
        batch = [
            make_transition(done=True),
            make_transition(done=False),
        ]
        data = collate_transitions(batch)
        assert data["dones"].dtype == torch.float32
        assert set(data["dones"].tolist()) <= {0.0, 1.0}

    def test_padding_with_zeros(self):
        """Shorter sequences are padded with token 0 (PAD_TOKEN_ID)."""
        batch = self._make_batch([5, 2])
        data = collate_transitions(batch)
        # Sequence 1 (len=2) should have pad tokens at positions 2,3,4
        padded_row = data["tokens"][1]
        assert padded_row[2].item() == 0
        assert padded_row[3].item() == 0
        assert padded_row[4].item() == 0

    def test_real_tokens_not_overwritten(self):
        """Non-padding positions retain their original token values."""
        trace = np.array([10, 20, 30], dtype=np.uint16)  # → tokens [11, 21, 31]
        next_trace = np.array([1, 2], dtype=np.uint16)
        from dl_seed_optimizer.tokenizer import tokenize_fixed
        t = Transition(
            seed_id=0,
            trace_tokens=tokenize_fixed(trace),
            trace_len=3,
            reward=0.0,
            next_trace_tokens=tokenize_fixed(next_trace),
            next_trace_len=2,
            done=False,
        )
        data = collate_transitions([t])
        assert data["tokens"][0, :3].tolist() == [11, 21, 31]

    def test_all_dtypes(self):
        """All output tensors have expected dtypes."""
        batch = self._make_batch([4, 4])
        data = collate_transitions(batch)
        assert data["tokens"].dtype == torch.long
        assert data["lengths"].dtype == torch.long
        assert data["rewards"].dtype == torch.float32
        assert data["dones"].dtype == torch.float32
        assert data["next_tokens"].dtype == torch.long
        assert data["next_lengths"].dtype == torch.long

    def test_variable_length_large_batch(self):
        """Batch of 32 with varying lengths (1..50) collates without error."""
        lengths = list(range(1, 33))
        batch = self._make_batch(lengths, lengths)
        data = collate_transitions(batch)
        assert data["tokens"].shape == (32, 32)  # max length is 32
        assert data["lengths"].max().item() == 32
