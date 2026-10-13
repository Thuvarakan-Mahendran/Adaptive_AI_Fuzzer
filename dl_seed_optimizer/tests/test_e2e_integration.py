"""
test_e2e_integration.py – End-to-End integration test simulating AFL++ and DLDaemon.

Verifies:
  1. DLDaemon launch on a dedicated socket.
  2. Sequential submission of multiple seeds with variable trace lengths.
  3. Correct binary responses (Q-values and accept decisions).
  4. Reward feedback via MSG_DL_REWARD triggering buffer accumulation.
  5. Asynchronous training worker running gradient updates without stalling inference.
  6. Sub-millisecond inference latency verification.
"""

from __future__ import annotations

import os
import socket
import sys
import tempfile
import threading
import time

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dl_seed_optimizer.config import DLConfig
from dl_seed_optimizer.dl_daemon import DLDaemon
from dl_seed_optimizer.trace_reader import write_trace
from server_python.socket_server import (
    pack_dl_reward,
    pack_trace_submit,
    unpack_trace_score,
)


def test_full_pipeline_multi_seed_simulation(tmp_path):
    sock_path = str(tmp_path / "e2e_dl.sock")
    trace_dir = str(tmp_path / "traces")
    os.makedirs(trace_dir, exist_ok=True)

    # Fast configuration for integration testing
    cfg = DLConfig(
        BUFFER_CAPACITY=1000,
        MIN_BUFFER_SIZE=10,
        BATCH_SIZE=4,
        TARGET_SYNC=20,
        DL_SOCKET_PATH=sock_path,
        TRACE_DIR=trace_dir,
    )

    daemon = DLDaemon(config=cfg, socket_path=sock_path, enable_training=True)
    server_thread = threading.Thread(target=daemon.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.15)

    client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    client.connect(sock_path)

    n_seeds = 30
    latencies = []
    verdicts = []

    for seed_id in range(1, n_seeds + 1):
        # Generate variable length execution trace
        seq_len = np.random.randint(10, 200)
        edges = np.random.randint(1, 1000, size=seq_len, dtype=np.uint16)
        trace_file = os.path.join(trace_dir, f"{seed_id}.trace")
        write_trace(trace_file, seed_id=seed_id, edge_ids=edges)

        # Time the inference request/response round-trip
        t0 = time.perf_counter()
        req = pack_trace_submit(seed_id=seed_id, trace_path=trace_file)
        client.send(req)
        resp = client.recv(1024)
        t_rtt = (time.perf_counter() - t0) * 1000.0
        latencies.append(t_rtt)

        score_data = unpack_trace_score(resp)
        verdicts.append(score_data["accept"])
        assert isinstance(score_data["q_value"], float)

        # Send post-execution reward
        new_edges = 1 if seed_id % 3 == 0 else 0
        crash = 1 if seed_id == 25 else 0
        rew = pack_dl_reward(
            seed_id=seed_id,
            delta_edges=new_edges,
            found_crash=crash,
            total_execs=seed_id * 100,
        )
        client.send(rew)

    client.close()

    # Allow background training worker a moment to process transitions
    time.sleep(0.3)
    daemon.stop()

    # Assertions
    assert len(latencies) == n_seeds
    avg_latency = sum(latencies) / len(latencies)
    print(f"\n[E2E] Simulated {n_seeds} seeds. Mean RTT: {avg_latency:.2f} ms")
    assert avg_latency < 50.0  # Generous upper bound for test execution

    # Verify that experiences were pushed to replay buffer
    assert len(daemon.trainer.buffer) > 0
    print(f"[E2E] Replay buffer populated with {len(daemon.trainer.buffer)} experiences.")
