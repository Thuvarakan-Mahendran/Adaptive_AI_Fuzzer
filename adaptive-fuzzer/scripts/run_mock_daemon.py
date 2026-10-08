#!/usr/bin/env python3
"""
scripts/run_mock_daemon.py – Mock PPO daemon for IPC integration testing.

Refactored to import the centralized AF_UNIX SOCK_SEQPACKET implementation
from adaptive-fuzzer/ipc_bridge/server_python.
"""

import argparse
import logging
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_DIR)
_IPC_DIR = os.path.join(_PROJECT_DIR, "ipc_bridge")
if _IPC_DIR not in sys.path:
    sys.path.insert(0, _IPC_DIR)

from server_python.mock_server import main

if __name__ == "__main__":
    main()
