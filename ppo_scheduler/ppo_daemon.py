#!/usr/bin/env python3
"""
ppo_daemon.py – Production PPO learning daemon bridging AFL++ and PPOAgent.

Imports:
  - ipc_bridge.server_python.IPCServer for SOCK_SEQPACKET communication.
  - ppo_scheduler.agent.PPOAgent for policy inference and gradient updates.

Non-blocking concurrency architecture:
  1. Incoming MSG_MUT_REQ is unpacked into a 192-D state vector (128 cov + 64 hist).
  2. PPOAgent generates actions immediately via select_action().
  3. MSG_MUT_RESP is transmitted back to the C mutator instantly (< 1 ms latency).
  4. If online learning is active, completed rollout buffers are passed to a
     background worker thread for optimization, guaranteeing that PyTorch
     backpropagation NEVER stalls the AFL++ forkserver.
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import signal
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure adaptive-fuzzer/ipc_bridge and Research are in sys.path
_PPO_DIR = os.path.dirname(os.path.abspath(__file__))
_RESEARCH_DIR = os.path.dirname(_PPO_DIR)
_IPC_DIR = os.path.join(_RESEARCH_DIR, "adaptive-fuzzer", "ipc_bridge")

for p in [_RESEARCH_DIR, _IPC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ppo_scheduler.agent import PPOAgent
from ppo_scheduler.buffer import RolloutBuffer
from ppo_scheduler.config import PPOConfig
from server_python.socket_server import (
    DEFAULT_SOCKET_PATH,
    MSG_MUT_REQ,
    MSG_SEED_QUERY,
    TOTAL_OPERATORS,
    IPCServer,
    pack_mut_resp,
    pack_seed_verd,
    unpack_mut_req,
    unpack_seed_query,
)

logger = logging.getLogger("ppo_daemon")


class BackgroundTrainer:
    """Asynchronous worker that updates PPOAgent in the background without blocking IPC."""

    def __init__(self, agent: PPOAgent) -> None:
        self.agent = agent
        self.update_queue: queue.Queue[RolloutBuffer] = queue.Queue(maxsize=4)
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def submit(self, buffer: RolloutBuffer) -> bool:
        try:
            self.update_queue.put_nowait(buffer)
            return True
        except queue.Full:
            logger.warning("Trainer queue full; skipping rollout update.")
            return False

    def _worker_loop(self) -> None:
        while self.running:
            try:
                buf = self.update_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                t0 = time.time()
                metrics = self.agent.update(buf)
                elapsed = (time.time() - t0) * 1000.0
                logger.info(
                    "Background PPO update completed in %.1f ms | Loss: %.4f | Entropy: %.4f",
                    elapsed, metrics.get("combined_loss", 0.0), metrics.get("entropy", 0.0)
                )
            except Exception as e:
                logger.error("Error during background PPO update: %s", e)
            finally:
                self.update_queue.task_done()

    def stop(self) -> None:
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


def decode_192d_state(coverage_vec_bytes: bytes, history_vec_floats: Tuple[float, ...], action_dim: int = 14) -> np.ndarray:
    """Decode MSG_MUT_REQ binary fields into a 192-D numpy state vector."""
    cov_bits = np.unpackbits(np.frombuffer(coverage_vec_bytes, dtype=np.uint8)).astype(np.float32)
    hist = np.array(history_vec_floats, dtype=np.float32)

    # Defensive check: ensure last_op slice (history[32:48]) is strictly all-zero or valid one-hot
    last_op = hist[32:48]
    ones = np.where(last_op == 1.0)[0]
    if not (np.all(last_op == 0.0) or (len(ones) == 1 and ones[0] < action_dim and np.sum(last_op) == 1.0)):
        hist[32:48] = 0.0

    return np.concatenate([cov_bits, hist])


def handle_client(
    conn: Any,
    agent: PPOAgent,
    config: PPOConfig,
    trainer: Optional[BackgroundTrainer],
    batch_size: int,
    verbose: bool,
) -> None:
    """Handle client session with immediate response delivery and deferred learning."""
    n_mut_reqs = 0
    n_seed_queries = 0

    buffer = RolloutBuffer(config) if trainer else None

    # Track last transition for credit assignment
    last_state: Optional[np.ndarray] = None
    last_action: Optional[int] = None
    last_log_prob: Optional[float] = None
    last_value: Optional[float] = None

    try:
        while True:
            try:
                data = conn.recv(4096)
            except OSError as e:
                if verbose:
                    logger.debug("recv() error: %s", e)
                break

            if not data:
                break

            msg_type = data[0]

            if msg_type == MSG_MUT_REQ:
                n_mut_reqs += 1
                req = unpack_mut_req(data)
                deferred_reward = req["deferred_reward"]
                queue_cur_changed = req["queue_cur_changed"]

                # 1. Decode current 192-D state
                state = decode_192d_state(req["coverage_vec"], req["history_vec"], config.ACTION_DIM)

                # 2. Select actions immediately
                # Primary action from current policy
                primary_action, log_prob, value = agent.select_action(state, deterministic=False)

                # Fill batch of K operations (primary action + additional policy samples)
                actions = [primary_action]
                for _ in range(batch_size - 1):
                    extra_act, _, _ = agent.select_action(state, deterministic=False)
                    actions.append(extra_act)

                # 3. Transmit MSG_MUT_RESP immediately (< 1 ms latency)
                resp = pack_mut_resp(actions, count=batch_size)
                conn.send(resp)

                # 4. Record transition into RolloutBuffer for training
                if trainer and buffer is not None and last_state is not None:
                    buffer.add(
                        state=last_state,
                        action=last_action,
                        reward=deferred_reward,
                        next_state=state,
                        value=last_value,
                        log_prob=last_log_prob,
                        done=(queue_cur_changed == 1),
                    )

                    if buffer.is_full():
                        trainer.submit(buffer)
                        buffer = RolloutBuffer(config)

                # Update state tracking
                last_state = state
                last_action = primary_action
                last_log_prob = log_prob
                last_value = value

            elif msg_type == MSG_SEED_QUERY:
                n_seed_queries += 1
                # Seed prioritization query – accept all with fixed Q-value for now
                resp = pack_seed_verd(q_value=0.85, accept=1)
                conn.send(resp)

            else:
                logger.warning("Unknown message type: 0x%02x (%d bytes)", msg_type, len(data))

    finally:
        conn.close()
        logger.info(
            "Session closed. Handled %d mutation batches and %d seed queries.",
            n_mut_reqs, n_seed_queries
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO Scheduler Daemon for AFL++ Custom Mutator")
    parser.add_argument("--sock", default=DEFAULT_SOCKET_PATH, help="UNIX domain socket path")
    parser.add_argument("--batch-size", type=int, default=8, help="Operator batch size per IPC response")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint to load/save")
    parser.add_argument("--no-train", action="store_true", help="Inference only mode (disable online training)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        format="[ppo_daemon %(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        level=log_level,
        stream=sys.stderr,
    )

    # Initialize PPOConfig for 14 operators
    config = PPOConfig(ACTION_DIM=TOTAL_OPERATORS, ROLLOUT_LENGTH=128)
    agent = PPOAgent(config)

    if args.checkpoint and os.path.exists(args.checkpoint):
        logger.info("Loading checkpoint from %s", args.checkpoint)
        agent.load(args.checkpoint)

    trainer = None
    if not args.no_train:
        trainer = BackgroundTrainer(agent)
        logger.info("Online training enabled with background worker.")

    # Warm up PyTorch policy network so the first client request experiences sub-millisecond latency
    dummy_state = np.zeros(config.STATE_DIM, dtype=np.float32)
    agent.select_action(dummy_state, deterministic=False)
    logger.info("PyTorch policy network warmed up.")

    server = IPCServer(socket_path=args.sock)

    def _shutdown(signum, frame):
        logger.info("Received signal %d, shutting down cleanly...", signum)
        if trainer:
            trainer.stop()
        if args.checkpoint:
            os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint)), exist_ok=True)
            agent.save(args.checkpoint)
            logger.info("Saved checkpoint to %s", args.checkpoint)
        server.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    server.start()
    logger.info("PPO Learning Daemon ready on %s", args.sock)

    while True:
        client_res = server.accept(timeout=1.0)
        if client_res is None:
            continue
        conn, addr = client_res
        logger.info("AFL++ instance connected.")
        handle_client(conn, agent, config, trainer, args.batch_size, args.verbose)


if __name__ == "__main__":
    main()
