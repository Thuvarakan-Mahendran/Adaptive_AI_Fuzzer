"""
socket_server.py – Reusable AF_UNIX SOCK_SEQPACKET server & packet codec.

Provides:
  - Exact binary packing / unpacking matching ipc_bridge/common/protocol.h.
  - IPCServer: Lifecycle management, stale-socket unlinking, connection handling.
"""

from __future__ import annotations

import logging
import os
import select
import signal
import socket
import struct
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------
DEFAULT_SOCKET_PATH = "/tmp/afl_ai_bridge.sock"
DL_SOCKET_PATH      = "/tmp/afl_dl_bridge.sock"

MSG_MUT_REQ      = 0x01
MSG_MUT_RESP     = 0x02
MSG_SEED_QUERY   = 0x03
MSG_SEED_VERD    = 0x04
MSG_TRACE_SUBMIT = 0x05
MSG_TRACE_SCORE  = 0x06
MSG_DL_REWARD    = 0x07

MUT_BATCH_MAX   = 32
TRACE_STEPS_MAX = 512
TRACE_PATH_MAX  = 256
TOTAL_OPERATORS = 14

# Struct formats (little-endian native __attribute__((packed)))
MUT_REQ_FMT   = "<B f 16s 64f B"
MUT_REQ_SIZE  = struct.calcsize(MUT_REQ_FMT)   # 278 bytes

MUT_RESP_FMT  = "<B B " + "H" * MUT_BATCH_MAX
MUT_RESP_SIZE = struct.calcsize(MUT_RESP_FMT)  # 66 bytes

SEED_QUERY_HEADER_FMT  = "<B I I"
SEED_QUERY_HEADER_SIZE = struct.calcsize(SEED_QUERY_HEADER_FMT) # 9 bytes
TRACE_STEP_FMT         = "<H B B"
TRACE_STEP_SIZE        = struct.calcsize(TRACE_STEP_FMT)         # 4 bytes
SEED_QUERY_FMT         = "<B I I" + "H B B" * TRACE_STEPS_MAX
SEED_QUERY_SIZE        = struct.calcsize(SEED_QUERY_FMT)        # 2057 bytes

SEED_VERD_FMT  = "<B f B"
SEED_VERD_SIZE = struct.calcsize(SEED_VERD_FMT) # 6 bytes

TRACE_SUBMIT_FMT  = f"<B I {TRACE_PATH_MAX}s"
TRACE_SUBMIT_SIZE = struct.calcsize(TRACE_SUBMIT_FMT) # 261 bytes

TRACE_SCORE_FMT  = "<B f B"
TRACE_SCORE_SIZE = struct.calcsize(TRACE_SCORE_FMT)   # 6 bytes

DL_REWARD_FMT  = "<B I I B I"
DL_REWARD_SIZE = struct.calcsize(DL_REWARD_FMT)       # 14 bytes

logger = logging.getLogger("ipc_bridge.server")


# ---------------------------------------------------------------------------
# Packet Serialization & Deserialization
# ---------------------------------------------------------------------------

def unpack_mut_req(data: bytes) -> Dict[str, Any]:
    """Unpack raw binary payload into a structured MSG_MUT_REQ dictionary."""
    if len(data) < MUT_REQ_SIZE:
        raise ValueError(f"Payload too short for MSG_MUT_REQ: {len(data)} < {MUT_REQ_SIZE}")

    fields = struct.unpack_from(MUT_REQ_FMT, data)
    msg_type = fields[0]
    deferred_reward = fields[1]
    coverage_vec = fields[2]
    history_vec = fields[3:67]
    queue_cur_changed = fields[67]

    return {
        "msg_type": msg_type,
        "deferred_reward": float(deferred_reward),
        "coverage_vec": coverage_vec,        # 16 bytes (128 bits)
        "history_vec": tuple(history_vec),   # 64 float32 values
        "queue_cur_changed": int(queue_cur_changed),
    }


def pack_mut_resp(action_ids: Sequence[int], count: Optional[int] = None) -> bytes:
    """Pack scheduled operator IDs into a 66-byte MSG_MUT_RESP datagram."""
    n = len(action_ids) if count is None else count
    n = min(n, MUT_BATCH_MAX)

    padded_actions = list(action_ids[:n]) + [0] * (MUT_BATCH_MAX - n)
    return struct.pack(MUT_RESP_FMT, MSG_MUT_RESP, n, *padded_actions)


def unpack_seed_query(data: bytes) -> Dict[str, Any]:
    """Unpack raw binary payload into a structured MSG_SEED_QUERY dictionary."""
    if len(data) < SEED_QUERY_HEADER_SIZE:
        raise ValueError(f"Payload too short for seed query header: {len(data)}")

    msg_type, seed_id, trace_len = struct.unpack_from(SEED_QUERY_HEADER_FMT, data)
    steps: List[Tuple[int, int, int]] = []

    actual_steps = min(trace_len, TRACE_STEPS_MAX)
    offset = SEED_QUERY_HEADER_SIZE
    for _ in range(actual_steps):
        if offset + TRACE_STEP_SIZE > len(data):
            break
        edge_id, hit_count, fault_flags = struct.unpack_from(TRACE_STEP_FMT, data, offset)
        steps.append((edge_id, hit_count, fault_flags))
        offset += TRACE_STEP_SIZE

    return {
        "msg_type": msg_type,
        "seed_id": seed_id,
        "trace_len": trace_len,
        "steps": steps,
    }


def pack_seed_verd(q_value: float, accept: int) -> bytes:
    """Pack seed verdict into a 6-byte MSG_SEED_VERD datagram."""
    return struct.pack(SEED_VERD_FMT, MSG_SEED_VERD, float(q_value), int(accept))


def unpack_trace_submit(data: bytes) -> Dict[str, Any]:
    """Unpack raw binary payload into a structured MSG_TRACE_SUBMIT dictionary."""
    if len(data) < TRACE_SUBMIT_SIZE:
        raise ValueError(f"Payload too short for MSG_TRACE_SUBMIT: {len(data)} < {TRACE_SUBMIT_SIZE}")

    msg_type, seed_id, trace_path_raw = struct.unpack_from(TRACE_SUBMIT_FMT, data)
    trace_path = trace_path_raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")

    return {
        "msg_type": msg_type,
        "seed_id": int(seed_id),
        "trace_path": trace_path,
    }


def pack_trace_submit(seed_id: int, trace_path: str) -> bytes:
    """Pack trace file notification into a 261-byte MSG_TRACE_SUBMIT datagram."""
    path_bytes = trace_path.encode("utf-8")[:TRACE_PATH_MAX]
    path_bytes_padded = path_bytes.ljust(TRACE_PATH_MAX, b"\x00")
    return struct.pack(TRACE_SUBMIT_FMT, MSG_TRACE_SUBMIT, int(seed_id), path_bytes_padded)


def unpack_trace_score(data: bytes) -> Dict[str, Any]:
    """Unpack raw binary payload into a structured MSG_TRACE_SCORE dictionary."""
    if len(data) < TRACE_SCORE_SIZE:
        raise ValueError(f"Payload too short for MSG_TRACE_SCORE: {len(data)} < {TRACE_SCORE_SIZE}")

    msg_type, q_value, accept = struct.unpack_from(TRACE_SCORE_FMT, data)
    return {
        "msg_type": msg_type,
        "q_value": float(q_value),
        "accept": int(accept),
    }


def pack_trace_score(q_value: float, accept: int) -> bytes:
    """Pack score verdict into a 6-byte MSG_TRACE_SCORE datagram."""
    return struct.pack(TRACE_SCORE_FMT, MSG_TRACE_SCORE, float(q_value), int(accept))


def unpack_dl_reward(data: bytes) -> Dict[str, Any]:
    """Unpack raw binary payload into a structured MSG_DL_REWARD dictionary."""
    if len(data) < DL_REWARD_SIZE:
        raise ValueError(f"Payload too short for MSG_DL_REWARD: {len(data)} < {DL_REWARD_SIZE}")

    msg_type, seed_id, delta_edges, found_crash, total_execs = struct.unpack_from(DL_REWARD_FMT, data)
    return {
        "msg_type": msg_type,
        "seed_id": int(seed_id),
        "delta_edges": int(delta_edges),
        "found_crash": int(found_crash),
        "total_execs": int(total_execs),
    }


def pack_dl_reward(seed_id: int, delta_edges: int, found_crash: int, total_execs: int) -> bytes:
    """Pack seed fuzzing outcome into a 14-byte MSG_DL_REWARD datagram."""
    return struct.pack(
        DL_REWARD_FMT,
        MSG_DL_REWARD,
        int(seed_id),
        int(delta_edges),
        int(found_crash),
        int(total_execs),
    )


# ---------------------------------------------------------------------------
# Reusable IPCServer
# ---------------------------------------------------------------------------

class IPCServer:
    """Reusable AF_UNIX SOCK_SEQPACKET server with non-blocking polling."""

    def __init__(self, socket_path: str = DEFAULT_SOCKET_PATH, timeout: Optional[float] = None) -> None:
        self.socket_path = socket_path
        self.timeout = timeout
        self.server_sock: Optional[socket.socket] = None
        self._running = False

    def start(self) -> None:
        """Clean up stale socket file and bind SEQPACKET server."""
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError as e:
                logger.warning("Failed to unlink stale socket: %s", e)

        self.server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.server_sock.bind(self.socket_path)
        self.server_sock.listen(4)
        self._running = True
        logger.info("IPCServer listening on %s", self.socket_path)

    def accept(self, timeout: Optional[float] = None) -> Optional[Tuple[socket.socket, Any]]:
        """Accept an incoming client connection with optional timeout."""
        if not self.server_sock or not self._running:
            return None

        effective_timeout = timeout if timeout is not None else self.timeout
        if effective_timeout is not None:
            r, _, _ = select.select([self.server_sock], [], [], effective_timeout)
            if not r:
                return None

        try:
            conn, addr = self.server_sock.accept()
            return conn, addr
        except OSError:
            return None

    def close(self) -> None:
        """Shut down server and remove socket file."""
        self._running = False
        if self.server_sock:
            try:
                self.server_sock.close()
            except OSError:
                pass
            self.server_sock = None

        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass
        logger.info("IPCServer closed.")

    def __enter__(self) -> IPCServer:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
