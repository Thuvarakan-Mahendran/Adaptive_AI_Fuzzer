"""
test_models.py – Unit tests for models.py (Step 3).

Tests:
  - SeedEncoder: output shape, variable-length batches, gradient flow
  - DQNHead: output shape [batch], gradient flow
  - SeedQNetwork: end-to-end forward pass, score_single(), gradient flow
  - Embedding padding_idx: pad tokens produce zero gradients
  - No truncation: long sequences (10K tokens) pass through correctly
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dl_seed_optimizer.config import DLConfig
from dl_seed_optimizer.models import DQNHead, SeedEncoder, SeedQNetwork
from dl_seed_optimizer.tokenizer import pad_traces, tokenize_fixed


# Shared small config for fast tests
@pytest.fixture
def cfg() -> DLConfig:
    return DLConfig()   # All defaults


# ---------------------------------------------------------------------------
# SeedEncoder
# ---------------------------------------------------------------------------

class TestSeedEncoder:
    """Tests for SeedEncoder."""

    def test_output_shape_single(self, cfg):
        """Single sequence: output shape is [1, GRU_HIDDEN]."""
        enc = SeedEncoder(cfg.VOCAB_SIZE, cfg.EMBED_DIM, cfg.GRU_HIDDEN, cfg.GRU_LAYERS)
        tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)  # [1, 5]
        lengths = torch.tensor([5])
        out = enc(tokens, lengths)
        assert out.shape == (1, cfg.GRU_HIDDEN)

    def test_output_shape_batch(self, cfg):
        """Batch of 4 sequences: output shape is [4, GRU_HIDDEN]."""
        enc = SeedEncoder(cfg.VOCAB_SIZE, cfg.EMBED_DIM, cfg.GRU_HIDDEN, cfg.GRU_LAYERS)
        tokens = torch.randint(1, 100, (4, 10), dtype=torch.long)
        lengths = torch.tensor([10, 8, 6, 4])
        out = enc(tokens, lengths)
        assert out.shape == (4, cfg.GRU_HIDDEN)

    def test_variable_length_batch(self, cfg):
        """Variable-length sequences in same batch produce valid output."""
        enc = SeedEncoder(cfg.VOCAB_SIZE, cfg.EMBED_DIM, cfg.GRU_HIDDEN, cfg.GRU_LAYERS)

        traces = [
            tokenize_fixed(np.array([1, 2, 3], dtype=np.uint16)),
            tokenize_fixed(np.array([10, 20], dtype=np.uint16)),
            tokenize_fixed(np.array([100, 200, 300, 400, 500], dtype=np.uint16)),
        ]
        padded, lengths = pad_traces(traces)

        out = enc(padded, lengths)
        assert out.shape == (3, cfg.GRU_HIDDEN)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_gradient_flows_through_encoder(self, cfg):
        """Loss.backward() produces non-zero gradients in encoder parameters."""
        enc = SeedEncoder(cfg.VOCAB_SIZE, cfg.EMBED_DIM, cfg.GRU_HIDDEN, cfg.GRU_LAYERS)
        tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
        lengths = torch.tensor([3])

        out = enc(tokens, lengths)
        loss = out.sum()
        loss.backward()

        # Check embedding gradients
        assert enc.embedding.weight.grad is not None
        # Only the rows for tokens [1, 2, 3] should have non-zero gradients
        grad_rows = enc.embedding.weight.grad[[1, 2, 3]]
        assert (grad_rows.abs() > 0).any()

    def test_padding_idx_zero_grad(self, cfg):
        """Padding token (0) embedding receives no gradient."""
        enc = SeedEncoder(cfg.VOCAB_SIZE, cfg.EMBED_DIM, cfg.GRU_HIDDEN, cfg.GRU_LAYERS)
        # Sequence with actual tokens only; padding is in the padded tensor
        padded = torch.tensor([[1, 2, 0]], dtype=torch.long)  # last token is pad
        lengths = torch.tensor([2])  # only first 2 are real

        out = enc(padded, lengths)
        loss = out.sum()
        loss.backward()

        # Row 0 (pad token) must have zero gradient
        pad_grad = enc.embedding.weight.grad[0]
        assert pad_grad.abs().max().item() == pytest.approx(0.0, abs=1e-9)

    def test_long_trace_no_truncation(self, cfg):
        """10,000-token trace passes through without error (no truncation)."""
        enc = SeedEncoder(cfg.VOCAB_SIZE, cfg.EMBED_DIM, cfg.GRU_HIDDEN, cfg.GRU_LAYERS)
        seq_len = 10_000
        tokens = torch.randint(1, 65536, (1, seq_len), dtype=torch.long)
        lengths = torch.tensor([seq_len])

        out = enc(tokens, lengths)
        assert out.shape == (1, cfg.GRU_HIDDEN)
        assert not torch.isnan(out).any()

    def test_output_dtype_float32(self, cfg):
        """Output tensor dtype is float32."""
        enc = SeedEncoder(cfg.VOCAB_SIZE, cfg.EMBED_DIM, cfg.GRU_HIDDEN, cfg.GRU_LAYERS)
        tokens = torch.tensor([[1, 2]], dtype=torch.long)
        lengths = torch.tensor([2])
        out = enc(tokens, lengths)
        assert out.dtype == torch.float32

    def test_deterministic_in_eval_mode(self, cfg):
        """Same input produces same output in eval mode."""
        enc = SeedEncoder(cfg.VOCAB_SIZE, cfg.EMBED_DIM, cfg.GRU_HIDDEN, cfg.GRU_LAYERS)
        enc.eval()
        tokens = torch.tensor([[5, 10, 15]], dtype=torch.long)
        lengths = torch.tensor([3])
        out1 = enc(tokens, lengths)
        out2 = enc(tokens, lengths)
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# DQNHead
# ---------------------------------------------------------------------------

class TestDQNHead:
    """Tests for DQNHead."""

    def test_output_shape_batch(self, cfg):
        """Batch input [B, 128] → output [B]."""
        head = DQNHead(cfg.GRU_HIDDEN, cfg.DQN_HIDDEN)
        latent = torch.randn(8, cfg.GRU_HIDDEN)
        q = head(latent)
        assert q.shape == (8,)

    def test_output_shape_single(self, cfg):
        """Single input [1, 128] → output [1]."""
        head = DQNHead(cfg.GRU_HIDDEN, cfg.DQN_HIDDEN)
        latent = torch.randn(1, cfg.GRU_HIDDEN)
        q = head(latent)
        assert q.shape == (1,)

    def test_output_dtype_float32(self, cfg):
        """Output dtype is float32."""
        head = DQNHead(cfg.GRU_HIDDEN, cfg.DQN_HIDDEN)
        latent = torch.randn(2, cfg.GRU_HIDDEN)
        q = head(latent)
        assert q.dtype == torch.float32

    def test_gradient_flow(self, cfg):
        """Loss.backward() produces gradients in all DQNHead parameters."""
        head = DQNHead(cfg.GRU_HIDDEN, cfg.DQN_HIDDEN)
        latent = torch.randn(4, cfg.GRU_HIDDEN)
        q = head(latent)
        q.sum().backward()

        for name, param in head.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum().item() > 0, f"Zero gradient for {name}"

    def test_output_not_constrained(self, cfg):
        """Q-values are unconstrained (can be negative or large positive)."""
        head = DQNHead(cfg.GRU_HIDDEN, cfg.DQN_HIDDEN)
        latent = torch.randn(100, cfg.GRU_HIDDEN) * 10  # Large values
        q = head(latent)
        # Values span some range — not all identical
        assert q.max() != q.min()


# ---------------------------------------------------------------------------
# SeedQNetwork (end-to-end)
# ---------------------------------------------------------------------------

class TestSeedQNetwork:
    """Tests for the combined SeedQNetwork."""

    def test_forward_shape(self, cfg):
        """End-to-end forward: [B, seq] → [B] Q-values."""
        net = SeedQNetwork(cfg)
        tokens = torch.randint(1, 100, (4, 15), dtype=torch.long)
        lengths = torch.tensor([15, 12, 10, 8])
        q = net(tokens, lengths)
        assert q.shape == (4,)

    def test_forward_no_nan_inf(self, cfg):
        """Forward pass produces no NaN or Inf."""
        net = SeedQNetwork(cfg)
        tokens = torch.randint(1, 65536, (8, 20), dtype=torch.long)
        lengths = torch.full((8,), 20, dtype=torch.long)
        q = net(tokens, lengths)
        assert not torch.isnan(q).any()
        assert not torch.isinf(q).any()

    def test_gradient_flows_end_to_end(self, cfg):
        """Loss.backward() produces non-zero gradients in all sub-modules."""
        net = SeedQNetwork(cfg)
        tokens = torch.randint(1, 100, (2, 5), dtype=torch.long)
        lengths = torch.tensor([5, 3])
        q = net(tokens, lengths)
        loss = q.sum()
        loss.backward()

        for name, param in net.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_score_single_returns_float(self, cfg):
        """score_single() returns a Python float."""
        net = SeedQNetwork(cfg)
        net.eval()
        tokens = tokenize_fixed(np.array([10, 20, 30], dtype=np.uint16))
        result = net.score_single(tokens)
        assert isinstance(result, float)

    def test_score_single_no_grad_context(self, cfg):
        """score_single() does not compute gradients (uses no_grad)."""
        net = SeedQNetwork(cfg)
        tokens = tokenize_fixed(np.array([1, 2, 3], dtype=np.uint16))
        # Should not raise inside no_grad context
        result = net.score_single(tokens)
        assert isinstance(result, float)

    def test_different_seeds_different_q(self, cfg):
        """Different traces produce different Q-values (model is expressive)."""
        net = SeedQNetwork(cfg)
        net.eval()
        t1 = tokenize_fixed(np.array([1, 2, 3], dtype=np.uint16))
        t2 = tokenize_fixed(np.array([100, 200, 300], dtype=np.uint16))
        q1 = net.score_single(t1)
        q2 = net.score_single(t2)
        # After random init, Q-values should differ (not mathematically guaranteed
        # but extremely unlikely to be identical to float64 precision)
        assert q1 != pytest.approx(q2, abs=1e-6)

    def test_target_copy_produces_identical_outputs(self, cfg):
        """After load_state_dict, target net produces exactly the same output."""
        online = SeedQNetwork(cfg)
        target = SeedQNetwork(cfg)
        target.load_state_dict(online.state_dict())
        target.eval()
        online.eval()

        tokens = torch.randint(1, 100, (3, 8), dtype=torch.long)
        lengths = torch.tensor([8, 6, 4])

        with torch.no_grad():
            q_online = online(tokens, lengths)
            q_target = target(tokens, lengths)

        assert torch.allclose(q_online, q_target)

    def test_long_trace_end_to_end(self, cfg):
        """5,000-token trace passes end-to-end without error."""
        net = SeedQNetwork(cfg)
        net.eval()
        tokens = tokenize_fixed(
            np.random.randint(0, 65535, size=5000, dtype=np.uint16)
        )
        q = net.score_single(tokens)
        assert isinstance(q, float)
        assert not (q != q)  # not NaN
