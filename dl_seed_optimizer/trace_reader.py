"""
trace_reader.py – Read AFL++ execution trace files written by the C mutator.

Trace file format (little-endian binary):
    Offset  Size     Field
    0       4 bytes  seed_id    (uint32_t)
    4       4 bytes  trace_len  (uint32_t, number of edge events)
    8       2×N bytes edge_ids[N] (uint16_t array)

Total file size: 8 + 2 * trace_len bytes.

Files are written by trace_collector.c to TRACE_DIR (/dev/shm/afl_dl_traces/).
The path convention is: <TRACE_DIR>/<seed_id>.trace

This module provides:
  - read_trace(path)           → (seed_id, edge_ids_array)
  - write_trace(path, ...)     → writes a trace file (for testing / offline use)
  - trace_path(seed_id, dir)   → canonical path for a given seed ID
  - list_available_traces(dir) → all seed IDs with trace files present
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Default trace directory (tmpfs for memory-speed I/O)
DEFAULT_TRACE_DIR: str = "/dev/shm/afl_dl_traces"

# Binary format constants
_HEADER_FMT: str = "<II"        # seed_id (uint32), trace_len (uint32)
_HEADER_SIZE: int = struct.calcsize(_HEADER_FMT)   # 8 bytes
_EDGE_DTYPE: np.dtype = np.dtype("<u2")             # uint16 little-endian


def trace_path(seed_id: int, trace_dir: str = DEFAULT_TRACE_DIR) -> str:
    """Return the canonical file path for a given seed ID's trace.

    Convention: <trace_dir>/<seed_id>.trace

    Args:
        seed_id:   AFL++ queue entry index (uint32).
        trace_dir: Directory containing trace files.

    Returns:
        Absolute path string.
    """
    return os.path.join(trace_dir, f"{seed_id}.trace")


def read_trace(path: str) -> Tuple[int, np.ndarray]:
    """Read an execution trace file and return (seed_id, edge_ids).

    Args:
        path: Path to the .trace file.

    Returns:
        Tuple of:
          - seed_id  (int): The AFL++ queue entry index embedded in the file.
          - edge_ids (np.ndarray, dtype=uint16): Array of edge IDs in the
                     order they were recorded. Shape: (trace_len,).

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError:        If file is malformed or too short.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trace file not found: {path}")

    file_size = os.path.getsize(path)
    if file_size < _HEADER_SIZE:
        raise ValueError(
            f"Trace file too small ({file_size} bytes); "
            f"expected at least {_HEADER_SIZE} header bytes: {path}"
        )

    with open(path, "rb") as f:
        header = f.read(_HEADER_SIZE)
        seed_id, trace_len = struct.unpack(_HEADER_FMT, header)

        expected_body = trace_len * 2  # 2 bytes per uint16
        body = f.read(expected_body)

    if len(body) < expected_body:
        raise ValueError(
            f"Trace body truncated: expected {expected_body} bytes "
            f"({trace_len} edges), got {len(body)} bytes: {path}"
        )

    edge_ids: np.ndarray = np.frombuffer(body[:expected_body], dtype=_EDGE_DTYPE).copy()
    return int(seed_id), edge_ids


def read_trace_for_seed(
    seed_id: int, trace_dir: str = DEFAULT_TRACE_DIR
) -> Optional[np.ndarray]:
    """Read the trace for a given seed ID, returning None if not found.

    Convenience wrapper around read_trace() that constructs the path
    automatically and returns None on FileNotFoundError.

    Args:
        seed_id:   AFL++ queue entry index.
        trace_dir: Directory containing trace files.

    Returns:
        np.ndarray of edge IDs (dtype=uint16), or None if trace file absent.
    """
    path = trace_path(seed_id, trace_dir)
    try:
        _, edges = read_trace(path)
        return edges
    except FileNotFoundError:
        return None


def write_trace(
    path: str,
    seed_id: int,
    edge_ids: np.ndarray,
) -> None:
    """Write an execution trace file in the canonical binary format.

    Used by the C trace_collector.c and also available here for
    offline testing and synthetic trace generation.

    Args:
        path:     Destination file path.
        seed_id:  AFL++ queue entry index (uint32).
        edge_ids: Array of edge IDs to record. Will be cast to uint16.
                  Values outside [0, 65535] will be truncated by cast.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    edges_u16 = edge_ids.astype(np.uint16)
    trace_len = len(edges_u16)

    with open(path, "wb") as f:
        f.write(struct.pack(_HEADER_FMT, seed_id, trace_len))
        f.write(edges_u16.tobytes())


def list_available_traces(trace_dir: str = DEFAULT_TRACE_DIR) -> List[int]:
    """Return a sorted list of seed IDs for which trace files exist.

    Args:
        trace_dir: Directory to search.

    Returns:
        Sorted list of integer seed IDs.
    """
    if not os.path.isdir(trace_dir):
        return []

    seed_ids: List[int] = []
    for fname in os.listdir(trace_dir):
        if fname.endswith(".trace"):
            stem = fname[: -len(".trace")]
            try:
                seed_ids.append(int(stem))
            except ValueError:
                pass  # Skip malformed filenames

    return sorted(seed_ids)


def ensure_trace_dir(trace_dir: str = DEFAULT_TRACE_DIR) -> None:
    """Create the trace directory if it does not already exist.

    Safe to call multiple times. Uses os.makedirs with exist_ok=True.

    Args:
        trace_dir: Directory path to create.
    """
    os.makedirs(trace_dir, exist_ok=True)
