#!/usr/bin/env bash
# Runner for the PPO learning daemon from adaptive-fuzzer/scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESEARCH_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
exec python3 "${RESEARCH_DIR}/ppo_scheduler/ppo_daemon.py" "$@"
