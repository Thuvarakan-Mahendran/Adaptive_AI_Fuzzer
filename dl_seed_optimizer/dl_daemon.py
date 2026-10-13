#!/usr/bin/env python3
"""
dl_daemon.py – Production DL Seed Prioritization Daemon for AFL++.

Architecture:
  - Listens on UNIX domain socket: DL_SOCKET_PATH (/tmp/afl_dl_bridge.sock)
  - IPC handler thread:
      1. Receives MSG_TRACE_SUBMIT (seed_id, trace_path) from AFL++ mutator.
      2. Reads execution trace from /dev/shm.
      3. Scores seed via SeedQNetwork forward pass (or cache).
      4. Decides accept/skip based on Q-ranking & ε-greedy exploration.
      5. Responds with MSG_TRACE_SCORE (q_value, accept) in < 2ms.
      6. Receives MSG_DL_REWARD and queues experiences into ReplayBuffer.
  - Background training thread:
      Runs DQNTrainer.train_step() asynchronously so PyTorch gradient
      updates never block the AFL++ forkserver or IPC responses.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure adaptive-fuzzer/ipc_bridge and Research are in sys.path
_DL_DIR = os.path.dirname(os.path.abspath(__file__))
_RESEARCH_DIR = os.path.dirname(_DL_DIR)
_IPC_DIR = os.path.join(_RESEARCH_DIR, "adaptive-fuzzer", "ipc_bridge")

for p in [_RESEARCH_DIR, _IPC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dl_seed_optimizer.config import DLConfig
from dl_seed_optimizer.models import SeedQNetwork
from dl_seed_optimizer.replay_buffer import Transition
from dl_seed_optimizer.reward import compute_dl_reward
from dl_seed_optimizer.tokenizer import tokenize_fixed
from dl_seed_optimizer.trace_reader import read_trace
from dl_seed_optimizer.trainer import DQNTrainer
from server_python.socket_server import (
    DL_SOCKET_PATH,
    MSG_DL_REWARD,
    MSG_TRACE_SUBMIT,
    IPCServer,
    pack_trace_score,
    unpack_dl_reward,
    unpack_trace_submit,
)

logger = logging.getLogger("dl_daemon")


class BackgroundDQNWorker:
    """Asynchronous background worker that runs DQN training steps."""

    def __init__(self, trainer: DQNTrainer, on_sync_callback: Optional[Any] = None) -> None:
        self.trainer = trainer
        self.on_sync_callback = on_sync_callback
        self.running = True
        self._last_model_version = trainer.model_version
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def _worker_loop(self) -> None:
        while self.running:
            try:
                metrics = self.trainer.train_step()
                if metrics:
                    if self.trainer.model_version > self._last_model_version:
                        self._last_model_version = self.trainer.model_version
                        if self.on_sync_callback:
                            self.on_sync_callback()
                    time.sleep(0.001)
                else:
                    time.sleep(0.02)
            except Exception as e:
                logger.error("Error during background DQN update: %s", e)
                time.sleep(0.05)

    def stop(self) -> None:
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


class DLDaemon:
    """Production server managing IPC inference and asynchronous training."""

    def __init__(
        self,
        config: Optional[DLConfig] = None,
        socket_path: str = DL_SOCKET_PATH,
        enable_training: bool = True,
        checkpoint_path: str = "",
    ) -> None:
        self.config = config or DLConfig()
        self.socket_path = socket_path
        self.trainer = DQNTrainer(self.config)
        self.checkpoint_path = checkpoint_path
        self.enable_training = enable_training

        # Q-value Cache: (seed_id, model_version) -> float
        self.q_cache: Dict[Tuple[int, int], float] = {}
        self.cache_lock = threading.Lock()

        # Ranking state: recent Q-values for percentile thresholding
        self.recent_q_values: List[float] = []
        self.ranking_lock = threading.Lock()

        # Seed trace memory for replay transitions: seed_id -> (tokens, length)
        self.seed_traces: Dict[int, Tuple[torch.LongTensor, int]] = {}
        self.last_seed_id: Optional[int] = None
        self.seed_traces_lock = threading.Lock()

        # IPC Server
        self.server = IPCServer(socket_path=self.socket_path)
        self.running = False

        # Load checkpoint if provided
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            logger.info("Loading checkpoint from %s", self.checkpoint_path)
            self.trainer.load_checkpoint(self.checkpoint_path)

        # Background trainer
        self.worker: Optional[BackgroundDQNWorker] = None
        if self.enable_training:
            self.worker = BackgroundDQNWorker(
                self.trainer, on_sync_callback=self._on_target_sync
            )
            logger.info("Asynchronous DQN live training enabled.")

        # Warm up neural network
        self._warmup()

    def _warmup(self) -> None:
        """Warm up the neural network so the first query runs with zero JIT lag."""
        dummy_tokens = torch.randint(1, 100, (1, 16), dtype=torch.long, device=self.trainer.device)
        dummy_lens = torch.tensor([16], dtype=torch.long, device=self.trainer.device)
        with torch.no_grad():
            self.trainer.online_net.eval()
            _ = self.trainer.online_net(dummy_tokens, dummy_lens)
        logger.info("SeedQNetwork warmed up.")

    def _on_target_sync(self) -> None:
        """Called when target network syncs and model_version bumps."""
        with self.cache_lock:
            self.q_cache.clear()
        logger.debug("Target network synced: cleared Q-value cache.")

    def should_accept(self, q_value: float) -> int:
        """Determine whether to accept or skip a seed based on Q-ranking & ε-greedy."""
        # ε-greedy exploration: accept with probability ε
        if np.random.rand() < self.trainer.epsilon:
            return 1

        with self.ranking_lock:
            self.recent_q_values.append(q_value)
            if len(self.recent_q_values) > 500:
                self.recent_q_values.pop(0)

            # If fewer than 10 samples, accept all
            if len(self.recent_q_values) < 10:
                return 1

            threshold = float(np.percentile(self.recent_q_values, 100.0 - self.config.ACCEPT_PERCENTILE))
            return 1 if q_value >= threshold else 0

    def score_trace(self, seed_id: int, edges: np.ndarray) -> float:
        """Score an execution trace and cache the result."""
        model_ver = self.trainer.model_version
        cache_key = (seed_id, model_ver)

        with self.cache_lock:
            if cache_key in self.q_cache:
                return self.q_cache[cache_key]

        if len(edges) == 0:
            q_val = 0.0
        else:
            tokens = tokenize_fixed(edges)
            with torch.no_grad():
                self.trainer.online_net.eval()
                q_val = self.trainer.online_net.score_single(tokens)

        with self.cache_lock:
            self.q_cache[cache_key] = q_val

        return q_val

    def handle_client(self, conn: Any) -> None:
        """Handle IPC connection with non-blocking message processing."""
        n_queries = 0
        n_rewards = 0

        try:
            while self.running:
                try:
                    data = conn.recv(4096)
                except OSError:
                    break

                if not data:
                    break

                msg_type = data[0]

                if msg_type == MSG_TRACE_SUBMIT:
                    n_queries += 1
                    req = unpack_trace_submit(data)
                    seed_id = req["seed_id"]
                    trace_path = req["trace_path"]

                    # Read trace from /dev/shm
                    edges: np.ndarray = np.zeros(0, dtype=np.uint16)
                    try:
                        _, edges = read_trace(trace_path)
                    except Exception as e:
                        logger.debug("Could not read trace at %s: %s", trace_path, e)

                    # Store trace for potential replay transition
                    tokens = tokenize_fixed(edges) if len(edges) > 0 else torch.zeros(0, dtype=torch.long)
                    with self.seed_traces_lock:
                        self.seed_traces[seed_id] = (tokens, len(edges))
                        prev_seed = self.last_seed_id
                        self.last_seed_id = seed_id

                    # Inference scoring
                    q_val = self.score_trace(seed_id, edges)
                    verdict = self.should_accept(q_val)

                    # Immediate response
                    resp = pack_trace_score(q_val, verdict)
                    conn.send(resp)

                elif msg_type == MSG_DL_REWARD:
                    n_rewards += 1
                    rew_msg = unpack_dl_reward(data)
                    seed_id = rew_msg["seed_id"]
                    delta_edges = rew_msg["delta_edges"]
                    found_crash = rew_msg["found_crash"]
                    total_execs = rew_msg["total_execs"]

                    # Compute reward
                    reward = compute_dl_reward(
                        prev_total_edges=0,
                        curr_total_edges=delta_edges,
                        prev_crashes=0,
                        curr_crashes=found_crash,
                        total_execs=total_execs,
                    )

                    # Add experience to replay buffer if we have previous trace
                    with self.seed_traces_lock:
                        if seed_id in self.seed_traces and len(self.seed_traces) >= 2:
                            curr_toks, curr_len = self.seed_traces[seed_id]
                            # Next trace is the most recent seed other than seed_id
                            next_id = self.last_seed_id if self.last_seed_id != seed_id else seed_id
                            next_toks, next_len = self.seed_traces.get(next_id, (curr_toks, curr_len))

                            transition = Transition(
                                seed_id=seed_id,
                                trace_tokens=curr_toks,
                                trace_len=curr_len,
                                reward=reward,
                                next_trace_tokens=next_toks,
                                next_trace_len=next_len,
                                done=(found_crash == 1),
                            )
                            self.trainer.buffer.push(transition)
                else:
                    logger.warning("Unknown message type: 0x%02x (%d bytes)", msg_type, len(data))

        finally:
            conn.close()
            logger.info("Client session ended. Handled %d trace queries, %d rewards.", n_queries, n_rewards)

    def serve_forever(self) -> None:
        """Start server and accept client connections."""
        self.server.start()
        self.running = True
        logger.info("DLDaemon listening on %s", self.socket_path)

        try:
            while self.running:
                client_res = self.server.accept(timeout=0.5)
                if client_res is None:
                    continue
                conn, _ = client_res
                logger.info("AFL++ instance connected to DL bridge.")
                self.handle_client(conn)
        finally:
            self.stop()

    def stop(self) -> None:
        """Clean shutdown."""
        self.running = False
        if self.worker:
            self.worker.stop()
        if self.checkpoint_path:
            self.trainer.save_checkpoint(self.checkpoint_path)
        self.server.close()
        logger.info("DLDaemon stopped cleanly.")


def main() -> None:
    parser = argparse.ArgumentParser(description="DL Seed Prioritization Daemon for AFL++")
    parser.add_argument("--sock", default=DL_SOCKET_PATH, help="UNIX domain socket path")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint file")
    parser.add_argument("--no-train", action="store_true", help="Disable online training (inference only)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        format="[dl_daemon %(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        level=log_level,
        stream=sys.stderr,
    )

    daemon = DLDaemon(
        socket_path=args.sock,
        enable_training=(not args.no_train),
        checkpoint_path=args.checkpoint,
    )

    def _shutdown(signum, frame):
        logger.info("Caught signal %d; shutting down cleanly...", signum)
        daemon.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    daemon.serve_forever()


if __name__ == "__main__":
    main()
