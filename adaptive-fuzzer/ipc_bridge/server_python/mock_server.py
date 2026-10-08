#!/usr/bin/env python3
"""
mock_server.py – Standalone mock PPO daemon for IPC benchmarking & test harnesses.

Uses ipc_bridge.server_python.IPCServer to serve MSG_MUT_REQ and MSG_SEED_QUERY.
Runs out-of-band to verify datagram serialization, latency, and throughput.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import signal
import sys
import time

# Ensure current directory is in path when run standalone
_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_DIR)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from socket_server import (
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

logger = logging.getLogger("mock_server")


def handle_client(conn, verbose: bool, benchmark: bool) -> None:
    """Handle one connected AFL++ / test client."""
    n_mut_reqs = 0
    n_seed_queries = 0
    start_time = time.time()

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
                if not benchmark:
                    try:
                        req = unpack_mut_req(data)
                        if verbose:
                            logger.debug(
                                "MSG_MUT_REQ #%d: reward=%.3f, q_cur_chg=%d",
                                n_mut_reqs, req["deferred_reward"], req["queue_cur_changed"]
                            )
                    except Exception as e:
                        logger.error("Failed to unpack mut_req: %s", e)

                # Return 8 random actions
                action_ids = [random.randrange(TOTAL_OPERATORS) for _ in range(8)]
                resp = pack_mut_resp(action_ids, count=8)
                conn.send(resp)

            elif msg_type == MSG_SEED_QUERY:
                n_seed_queries += 1
                if not benchmark:
                    try:
                        query = unpack_seed_query(data)
                        if verbose:
                            logger.debug("MSG_SEED_QUERY #%d: seed_id=%d", n_seed_queries, query["seed_id"])
                    except Exception as e:
                        logger.error("Failed to unpack seed_query: %s", e)

                resp = pack_seed_verd(q_value=0.85, accept=1)
                conn.send(resp)

            else:
                logger.warning("Unknown message type: 0x%02x (%d bytes)", msg_type, len(data))

    finally:
        elapsed = time.time() - start_time
        conn.close()
        logger.info(
            "Client disconnected. Requests: %d mut_req, %d seed_queries (%.2f s, %.0f req/s)",
            n_mut_reqs, n_seed_queries, elapsed, (n_mut_reqs + n_seed_queries) / max(elapsed, 1e-6)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock PPO Server for IPC benchmarking")
    parser.add_argument("--sock", default=DEFAULT_SOCKET_PATH, help="Path to UNIX domain socket")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")
    parser.add_argument("--benchmark", action="store_true", help="High-throughput benchmark mode (minimal logging/parsing)")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else (logging.WARNING if args.benchmark else logging.INFO)
    logging.basicConfig(
        format="[mock_server %(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        level=log_level,
        stream=sys.stderr,
    )

    server = IPCServer(socket_path=args.sock)

    def _shutdown(signum, frame):
        logger.info("Signal %d received, shutting down...", signum)
        server.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    server.start()
    logger.info("Mock server running on %s", args.sock)

    while True:
        client_res = server.accept(timeout=1.0)
        if client_res is None:
            continue
        conn, addr = client_res
        logger.info("Client connected: %s", addr)
        handle_client(conn, args.verbose, args.benchmark)


if __name__ == "__main__":
    main()
