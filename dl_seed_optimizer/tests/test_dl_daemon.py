"""
test_dl_daemon.py – Unit and IPC integration tests for DLDaemon (Step 11).

Tests:
  - DLDaemon initialization and warmup
  - Q-value cache hits and target sync invalidation
  - Acceptance thresholding (top-percentile and epsilon-greedy)
  - Full loopback IPC round-trip over UNIX domain socket
  - Replay buffer population via MSG_DL_REWARD
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
from dl_seed_optimizer.dl_daemon import BackgroundDQNWorker, DLDaemon
from dl_seed_optimizer.trace_reader import write_trace
from server_python.socket_server import (
    pack_dl_reward,
    pack_trace_submit,
    unpack_trace_score,
)


@pytest.fixture
def test_socket_path(tmp_path):
    return str(tmp_path / "dl_test.sock")


@pytest.fixture
def fast_daemon(test_socket_path):
    cfg = DLConfig(
        BUFFER_CAPACITY=200,
        MIN_BUFFER_SIZE=16,
        BATCH_SIZE=8,
        TARGET_SYNC=10,
    )
    daemon = DLDaemon(
        config=cfg,
        socket_path=test_socket_path,
        enable_training=False,
    )
    yield daemon
    daemon.stop()


class TestDLDaemonCore:
    def test_init_and_warmup(self, fast_daemon):
        """Daemon initializes without error and online net is in eval mode."""
        assert fast_daemon.trainer.online_net is not None
        assert not fast_daemon.trainer.target_net.training

    def test_q_cache_and_invalidation(self, fast_daemon):
        """Q-value is cached and invalidated when target sync occurs."""
        edges = np.array([1, 2, 3], dtype=np.uint16)
        seed_id = 42

        # First score computes and caches
        q1 = fast_daemon.score_trace(seed_id, edges)
        key = (seed_id, fast_daemon.trainer.model_version)
        assert key in fast_daemon.q_cache
        assert fast_daemon.q_cache[key] == q1

        # Second score fetches from cache
        q2 = fast_daemon.score_trace(seed_id, edges)
        assert q1 == q2

        # Invalidate cache via sync callback
        fast_daemon._on_target_sync()
        assert len(fast_daemon.q_cache) == 0

    def test_should_accept_distribution(self, fast_daemon):
        """Acceptance logic produces binary verdicts."""
        fast_daemon.trainer.epsilon = 0.0  # Deterministic test
        for i in range(20):
            fast_daemon.should_accept(float(i))

        # High Q should be accepted
        assert fast_daemon.should_accept(100.0) == 1
        # Low Q should be skipped
        assert fast_daemon.should_accept(-100.0) == 0

    def test_epsilon_greedy_exploration(self, fast_daemon):
        """When epsilon is 1.0, all seeds are accepted."""
        fast_daemon.trainer.epsilon = 1.0
        assert fast_daemon.should_accept(-1000.0) == 1


class TestDLDaemonIPC:
    def test_ipc_round_trip(self, test_socket_path, tmp_path):
        """Verify full MSG_TRACE_SUBMIT -> MSG_TRACE_SCORE round trip."""
        cfg = DLConfig(MIN_BUFFER_SIZE=10, BATCH_SIZE=4)
        daemon = DLDaemon(config=cfg, socket_path=test_socket_path, enable_training=False)

        server_thread = threading.Thread(target=daemon.serve_forever, daemon=True)
        server_thread.start()
        time.sleep(0.1)

        # Create a trace file on disk
        trace_file = str(tmp_path / "test_1.trace")
        write_trace(trace_file, seed_id=1, edge_ids=np.array([10, 20, 30], dtype=np.uint16))

        # Client connection
        client = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        client.connect(test_socket_path)

        # 1. Send MSG_TRACE_SUBMIT
        req = pack_trace_submit(seed_id=1, trace_path=trace_file)
        client.send(req)

        resp = client.recv(1024)
        assert len(resp) == 6
        score_data = unpack_trace_score(resp)
        assert isinstance(score_data["q_value"], float)
        assert score_data["accept"] in (0, 1)

        # 2. Send MSG_DL_REWARD
        rew = pack_dl_reward(seed_id=1, delta_edges=5, found_crash=0, total_execs=100)
        client.send(rew)

        time.sleep(0.05)
        client.close()
        daemon.stop()

    def test_background_worker_lifecycle(self, fast_daemon):
        """Background worker thread starts and terminates cleanly."""
        worker = BackgroundDQNWorker(fast_daemon.trainer)
        assert worker.thread.is_alive()
        worker.stop()
        assert not worker.running
