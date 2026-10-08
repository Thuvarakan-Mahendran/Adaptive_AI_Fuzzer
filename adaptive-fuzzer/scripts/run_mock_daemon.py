#!/usr/bin/env python3
"""
scripts/run_mock_daemon.py – Mock PPO daemon for IPC integration testing.

Listens on a UNIX SOCK_SEQPACKET socket at /tmp/afl_ai_bridge.sock and
responds to the two message types used by libmutator.so:

  MSG_MUT_REQ  (0x01) → MSG_MUT_RESP  (0x02)
    Receive:  { msg_type, coverage_vec[16], history_vec[64], n_ops }
    Respond:  { msg_type=2, n_ops, op_ids[MUT_BATCH_MAX=8] }
    Policy:   Return 8 operator IDs sampled uniformly from [0, 13].
              This mirrors what a trained PPO agent would do but without
              the actual neural network inference.

  MSG_SEED_QUERY (0x03) → MSG_SEED_VERD (0x04)
    Receive:  { msg_type, seed_id, q_value (float32) }
    Respond:  { msg_type=4, seed_id, accept (uint8), q_value (float32) }
    Policy:   Accept all seeds (accept=1) with fixed q_value=0.85.

Wire format: all integers are little-endian, matching C __attribute__((packed)) structs.
Python struct format reference (per protocol.h, verified with sizeof()):
  MUT_REQ_FMT  = '<B f 16s 64f B'    # 278 bytes: type,reward,cov[16],hist[64],qchg
  MUT_RESP_FMT = '<B B' + 'H'*32     #  66 bytes: type,count,actions[32]
  SEED_VERD_FMT= '<B f B'            #   6 bytes: type,q_value,accept
  SEED_QUERY_FMT (recv only, 2057 bytes: type,seed_id,trace_len,steps[512])

Usage:
    python3 scripts/run_mock_daemon.py [--sock /tmp/afl_ai_bridge.sock] [-v]

Terminate with Ctrl-C or SIGTERM.
"""

import argparse
import os
import random
import socket
import struct
import sys
import signal
import logging

# ---------------------------------------------------------------------------
# Wire-protocol constants (must match protocol.h)
# ---------------------------------------------------------------------------
MSG_MUT_REQ   = 0x01
MSG_MUT_RESP  = 0x02
MSG_SEED_QUERY = 0x03
MSG_SEED_VERD  = 0x04

TOTAL_OPERATORS = 14        # TOTAL_OPERATORS in operators.h
MUT_BATCH_MAX   = 32        # MUT_BATCH_MAX in protocol.h (=32, not 8)
MUTATOR_BATCH_SIZE = 8      # Default batch size requested by mutator.c
DEFAULT_SOCK    = "/tmp/afl_ai_bridge.sock"

# ---- Verified struct sizes (clang-14, __attribute__((packed))) ----
# mut_request_msg (278 bytes):
#   uint8_t  type            [1]
#   float    deferred_reward [4]
#   uint8_t  coverage_vec[16][16]
#   float    history_vec[64][256]
#   uint8_t  queue_cur_changed [1]
MUT_REQ_FMT  = "<B f 16s 64f B"
MUT_REQ_SIZE = struct.calcsize(MUT_REQ_FMT)   # 278

# mut_response_msg (66 bytes):
#   uint8_t  type    [1]
#   uint8_t  count   [1]
#   uint16_t actions[32] [64]
MUT_RESP_FMT  = "<B B " + "H" * MUT_BATCH_MAX
MUT_RESP_SIZE = struct.calcsize(MUT_RESP_FMT)  # 66

# seed_query_msg (2057 bytes):
#   uint8_t   type       [1]
#   uint32_t  seed_id    [4]
#   uint32_t  trace_len  [4]
#   trace_step steps[512] [512*4=2048]  each: uint16_t + uint8_t + uint8_t
SEED_QUERY_FMT  = "<B I I" + "H B B" * 512
SEED_QUERY_SIZE = struct.calcsize(SEED_QUERY_FMT)  # 2057

# seed_verdict_msg (6 bytes):
#   uint8_t  type    [1]
#   float    q_value [4]
#   uint8_t  accept  [1]
SEED_VERD_FMT  = "<B f B"
SEED_VERD_SIZE = struct.calcsize(SEED_VERD_FMT)    # 6


# ---------------------------------------------------------------------------
# Daemon logic
# ---------------------------------------------------------------------------

def handle_client(conn: socket.socket, addr, verbose: bool) -> None:
    """Handle one connected client (libmutator.so instance) until disconnect."""
    log = logging.getLogger("mock_daemon")
    log.info("Client connected: %s", addr)
    n_reqs = 0
    n_queries = 0

    try:
        while True:
            # SEQPACKET delivers exactly one message per recv()
            try:
                data = conn.recv(4096)
            except OSError as e:
                log.debug("recv() error: %s", e)
                break

            if not data:
                log.info("Client disconnected (EOF)")
                break

            msg_type = data[0]

            if msg_type == MSG_MUT_REQ:
                n_reqs += 1
                if verbose:
                    log.debug("MSG_MUT_REQ #%d (%d bytes)", n_reqs, len(data))

                # Parse request (tolerate short payloads – use defaults)
                deferred_reward = 0.0
                queue_changed   = 0
                if len(data) >= MUT_REQ_SIZE:
                    fields = struct.unpack_from(MUT_REQ_FMT, data)
                    # fields: (type, deferred_reward, coverage_vec, h0..h63, queue_cur_changed)
                    deferred_reward = float(fields[1])
                    queue_changed   = int(fields[-1])

                # Reply with MUTATOR_BATCH_SIZE random operator IDs.
                # Pad the actions[] array to MUT_BATCH_MAX (32) as required by
                # mut_response_msg.  Unused slots are set to 0.
                n_ops  = MUTATOR_BATCH_SIZE
                op_ids = [random.randrange(TOTAL_OPERATORS) for _ in range(n_ops)]
                # Pad to MUT_BATCH_MAX entries
                op_ids_padded = op_ids + [0] * (MUT_BATCH_MAX - n_ops)

                resp = struct.pack(MUT_RESP_FMT, MSG_MUT_RESP, n_ops, *op_ids_padded)
                conn.send(resp)

                if verbose:
                    log.debug("  → MSG_MUT_RESP count=%d ops=%s reward=%.3f qchg=%d",
                              n_ops, op_ids, deferred_reward, queue_changed)

            elif msg_type == MSG_SEED_QUERY:
                n_queries += 1
                seed_id  = 0
                if len(data) >= 9:   # at least type(1)+seed_id(4)+trace_len(4)
                    _, seed_id, trace_len = struct.unpack_from("<B I I", data)
                if verbose:
                    log.debug("MSG_SEED_QUERY #%d seed_id=%d", n_queries, seed_id)

                # Accept all seeds unconditionally with fixed Q-value estimate
                # seed_verdict_msg: { type(1), q_value(4), accept(1) }
                resp = struct.pack(SEED_VERD_FMT, MSG_SEED_VERD, 0.85, 1)
                conn.send(resp)

                if verbose:
                    log.debug("  → MSG_SEED_VERD accept=1 q=0.85")

            else:
                log.warning("Unknown msg_type=0x%02x (%d bytes); ignoring", msg_type, len(data))

    finally:
        conn.close()
        log.info("Session closed (reqs=%d queries=%d)", n_reqs, n_queries)


def run_server(sock_path: str, verbose: bool) -> None:
    """Create SEQPACKET server socket and accept clients in a loop."""
    log = logging.getLogger("mock_daemon")

    # Remove stale socket file
    if os.path.exists(sock_path):
        os.unlink(sock_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    server.bind(sock_path)
    server.listen(4)
    log.info("Listening on %s (SOCK_SEQPACKET)", sock_path)
    log.info("Press Ctrl-C to stop.")

    def _shutdown(sig, frame):
        log.info("Signal %d received – shutting down.", sig)
        server.close()
        if os.path.exists(sock_path):
            os.unlink(sock_path)
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        try:
            conn, addr = server.accept()
        except OSError:
            break
        # Handle client synchronously (one AFL++ instance at a time).
        # For multi-instance fuzzing, replace with threading.Thread.
        handle_client(conn, addr, verbose)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Mock PPO daemon for AFL++ custom mutator IPC testing."
    )
    parser.add_argument(
        "--sock", default=DEFAULT_SOCK,
        help=f"UNIX socket path (default: {DEFAULT_SOCK})"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose per-message debug logging"
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        format="[mock_daemon %(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stderr,
    )

    run_server(args.sock, args.verbose)


if __name__ == "__main__":
    main()
