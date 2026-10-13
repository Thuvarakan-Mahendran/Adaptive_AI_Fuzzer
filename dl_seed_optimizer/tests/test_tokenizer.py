"""
test_tokenizer.py – Unit tests for tokenizer.py and trace_reader.py (Steps 2 & 7).

Tests:
  - tokenize_fixed: correct token IDs, shape, dtype, edge cases
  - detokenize_fixed: round-trip inverse of tokenize_fixed
  - pad_traces: correct padding and length extraction
  - write_trace / read_trace: round-trip binary file I/O
  - read_trace: error handling for missing/malformed files
  - list_available_traces: directory scanning
  - trace_path: canonical path construction
"""

from __future__ import annotations

import os
import struct
import tempfile

import numpy as np
import pytest
import torch

# Adjust sys.path so tests can be run from any directory
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dl_seed_optimizer.tokenizer import (
    PAD_TOKEN_ID,
    VOCAB_SIZE,
    detokenize_fixed,
    pad_traces,
    tokenize_fixed,
)
from dl_seed_optimizer.trace_reader import (
    DEFAULT_TRACE_DIR,
    ensure_trace_dir,
    list_available_traces,
    read_trace,
    read_trace_for_seed,
    trace_path,
    write_trace,
)


# =============================================================================
# tokenizer.py tests
# =============================================================================

class TestTokenizeFixed:
    """Tests for tokenize_fixed()."""

    def test_basic_mapping(self):
        """edge_id + 1 = token_id."""
        edges = np.array([0, 100, 65535], dtype=np.uint16)
        tokens = tokenize_fixed(edges)
        assert tokens.tolist() == [1, 101, 65536]

    def test_output_dtype(self):
        """Output must be torch.long (int64)."""
        edges = np.array([1, 2, 3], dtype=np.uint16)
        tokens = tokenize_fixed(edges)
        assert tokens.dtype == torch.long

    def test_output_shape(self):
        """Output shape matches input length."""
        edges = np.arange(50, dtype=np.uint16)
        tokens = tokenize_fixed(edges)
        assert tokens.shape == (50,)

    def test_empty_input(self):
        """Empty edge array → empty token tensor."""
        edges = np.array([], dtype=np.uint16)
        tokens = tokenize_fixed(edges)
        assert tokens.shape == (0,)
        assert tokens.dtype == torch.long

    def test_min_edge_id(self):
        """Minimum edge ID (0) maps to token 1 (not pad=0)."""
        tokens = tokenize_fixed(np.array([0], dtype=np.uint16))
        assert tokens[0].item() == 1

    def test_max_edge_id(self):
        """Maximum edge ID (65535) maps to token 65536 (within VOCAB_SIZE-1)."""
        tokens = tokenize_fixed(np.array([65535], dtype=np.uint16))
        assert tokens[0].item() == 65536
        assert tokens[0].item() < VOCAB_SIZE

    def test_pad_token_never_emitted(self):
        """tokenize_fixed never produces PAD_TOKEN_ID=0 for valid edge IDs."""
        edges = np.arange(0, 65536, dtype=np.uint16)  # All possible edges
        tokens = tokenize_fixed(edges)
        assert (tokens == PAD_TOKEN_ID).sum().item() == 0

    def test_invalid_negative_edge(self):
        """Negative edge IDs raise ValueError."""
        with pytest.raises(ValueError):
            tokenize_fixed(np.array([-1], dtype=np.int64))

    def test_invalid_large_edge(self):
        """Edge IDs > 65535 raise ValueError."""
        with pytest.raises(ValueError):
            tokenize_fixed(np.array([65536], dtype=np.int64))

    def test_int64_input(self):
        """Accepts int64 input (common from numpy operations)."""
        edges = np.array([10, 20, 30], dtype=np.int64)
        tokens = tokenize_fixed(edges)
        assert tokens.tolist() == [11, 21, 31]


class TestDetokenizeFixed:
    """Tests for detokenize_fixed()."""

    def test_round_trip(self):
        """tokenize → detokenize restores original edge IDs."""
        original = np.array([0, 100, 65535], dtype=np.uint16)
        tokens = tokenize_fixed(original)
        recovered = detokenize_fixed(tokens)
        np.testing.assert_array_equal(recovered, original.astype(np.int64))

    def test_pad_token_maps_to_minus_one(self):
        """PAD_TOKEN_ID (0) maps to -1 after detokenization."""
        tokens = torch.tensor([PAD_TOKEN_ID], dtype=torch.long)
        result = detokenize_fixed(tokens)
        assert result[0] == -1


class TestPadTraces:
    """Tests for pad_traces()."""

    def test_lengths_correct(self):
        """Returned lengths match input sequence lengths."""
        t1 = torch.tensor([1, 2, 3])
        t2 = torch.tensor([4, 5])
        _, lengths = pad_traces([t1, t2])
        assert lengths.tolist() == [3, 2]

    def test_padding_value_zero(self):
        """Shorter sequences are right-padded with PAD_TOKEN_ID=0."""
        t1 = torch.tensor([1, 2, 3])
        t2 = torch.tensor([4, 5])
        padded, _ = pad_traces([t1, t2])
        assert padded.shape == (2, 3)
        assert padded[1, 2].item() == PAD_TOKEN_ID

    def test_same_length(self):
        """No padding needed when all sequences have equal length."""
        t1 = torch.tensor([1, 2])
        t2 = torch.tensor([3, 4])
        padded, lengths = pad_traces([t1, t2])
        assert padded.shape == (2, 2)
        assert lengths.tolist() == [2, 2]

    def test_single_sequence(self):
        """Single-element list works correctly."""
        t1 = torch.tensor([5, 6, 7])
        padded, lengths = pad_traces([t1])
        assert padded.shape == (1, 3)
        assert lengths.tolist() == [3]

    def test_output_dtype_long(self):
        """Padded tensor dtype is torch.long."""
        padded, _ = pad_traces([torch.tensor([1, 2])])
        assert padded.dtype == torch.long


# =============================================================================
# trace_reader.py tests
# =============================================================================

class TestWriteReadTrace:
    """Tests for write_trace() / read_trace() round-trip."""

    def test_round_trip_basic(self, tmp_path):
        """Written trace is read back identically."""
        path = str(tmp_path / "0.trace")
        seed_id = 42
        edges = np.array([10, 200, 65535], dtype=np.uint16)

        write_trace(path, seed_id, edges)
        recovered_seed, recovered_edges = read_trace(path)

        assert recovered_seed == seed_id
        np.testing.assert_array_equal(recovered_edges, edges)

    def test_empty_trace(self, tmp_path):
        """Zero-length trace is handled correctly."""
        path = str(tmp_path / "empty.trace")
        write_trace(path, seed_id=0, edge_ids=np.array([], dtype=np.uint16))
        sid, edges = read_trace(path)
        assert sid == 0
        assert len(edges) == 0

    def test_large_trace(self, tmp_path):
        """Large trace (10,000 events) round-trips correctly."""
        path = str(tmp_path / "large.trace")
        edges = np.random.randint(0, 65536, size=10_000, dtype=np.uint16)
        write_trace(path, seed_id=999, edge_ids=edges)
        sid, recovered = read_trace(path)
        assert sid == 999
        assert len(recovered) == 10_000
        np.testing.assert_array_equal(recovered, edges)

    def test_seed_id_embedded_in_file(self, tmp_path):
        """seed_id in the file header must match what was written."""
        path = str(tmp_path / "check.trace")
        write_trace(path, seed_id=12345, edge_ids=np.array([1, 2], dtype=np.uint16))
        sid, _ = read_trace(path)
        assert sid == 12345

    def test_edge_dtype_preserved(self, tmp_path):
        """Recovered edges have uint16 dtype."""
        path = str(tmp_path / "dtype.trace")
        edges = np.array([0, 1, 2], dtype=np.uint16)
        write_trace(path, seed_id=0, edge_ids=edges)
        _, recovered = read_trace(path)
        assert recovered.dtype == np.uint16

    def test_file_size_is_correct(self, tmp_path):
        """File size matches 8 + 2*trace_len bytes."""
        path = str(tmp_path / "size.trace")
        n = 50
        edges = np.zeros(n, dtype=np.uint16)
        write_trace(path, seed_id=0, edge_ids=edges)
        assert os.path.getsize(path) == 8 + 2 * n


class TestReadTraceErrors:
    """Error handling tests for read_trace()."""

    def test_missing_file_raises(self, tmp_path):
        """FileNotFoundError for non-existent path."""
        with pytest.raises(FileNotFoundError):
            read_trace(str(tmp_path / "nonexistent.trace"))

    def test_truncated_file_raises(self, tmp_path):
        """ValueError for file with too-small header."""
        path = str(tmp_path / "bad.trace")
        with open(path, "wb") as f:
            f.write(b"\x00\x00")  # Only 2 bytes, need 8
        with pytest.raises(ValueError, match="too small"):
            read_trace(path)

    def test_truncated_body_raises(self, tmp_path):
        """ValueError when body is shorter than declared trace_len."""
        path = str(tmp_path / "trunc.trace")
        with open(path, "wb") as f:
            # Header says trace_len=100, but we write only 4 bytes of body
            f.write(struct.pack("<II", 0, 100))
            f.write(b"\x00" * 4)
        with pytest.raises(ValueError, match="truncated"):
            read_trace(path)


class TestReadTraceForSeed:
    """Tests for read_trace_for_seed()."""

    def test_returns_none_when_missing(self, tmp_path):
        """Returns None when seed trace doesn't exist."""
        result = read_trace_for_seed(999, trace_dir=str(tmp_path))
        assert result is None

    def test_returns_array_when_present(self, tmp_path):
        """Returns np.ndarray when trace file is present."""
        edges = np.array([1, 2, 3], dtype=np.uint16)
        path = trace_path(7, str(tmp_path))
        write_trace(path, seed_id=7, edge_ids=edges)
        result = read_trace_for_seed(7, trace_dir=str(tmp_path))
        assert result is not None
        np.testing.assert_array_equal(result, edges)


class TestTracePath:
    """Tests for trace_path()."""

    def test_path_format(self):
        """Path follows <dir>/<seed_id>.trace convention."""
        p = trace_path(42, "/some/dir")
        assert p == "/some/dir/42.trace"

    def test_default_dir(self):
        """Uses DEFAULT_TRACE_DIR when no dir specified."""
        p = trace_path(0)
        assert p.startswith(DEFAULT_TRACE_DIR)
        assert p.endswith("/0.trace")


class TestListAvailableTraces:
    """Tests for list_available_traces()."""

    def test_empty_dir(self, tmp_path):
        """Returns empty list for empty directory."""
        result = list_available_traces(str(tmp_path))
        assert result == []

    def test_nonexistent_dir(self, tmp_path):
        """Returns empty list for non-existent directory."""
        result = list_available_traces(str(tmp_path / "nope"))
        assert result == []

    def test_finds_trace_files(self, tmp_path):
        """Correctly identifies and sorts seed IDs from .trace files."""
        for sid in [10, 3, 7]:
            p = str(tmp_path / f"{sid}.trace")
            write_trace(p, seed_id=sid, edge_ids=np.array([1], dtype=np.uint16))
        result = list_available_traces(str(tmp_path))
        assert result == [3, 7, 10]

    def test_ignores_non_trace_files(self, tmp_path):
        """Non-.trace files are not included."""
        (tmp_path / "README.txt").write_text("hello")
        (tmp_path / "5.trace").write_bytes(
            struct.pack("<II", 5, 0)
        )
        result = list_available_traces(str(tmp_path))
        assert result == [5]
