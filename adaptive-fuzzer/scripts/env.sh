#!/usr/bin/env bash
# =============================================================================
# scripts/env.sh – Environment setup for running AFL++ with the AI mutator.
#
# Usage:
#   source scripts/env.sh          # from adaptive-fuzzer/ root
#   source adaptive-fuzzer/scripts/env.sh   # from Research/
#
# After sourcing, run AFL++ normally:
#   afl-fuzz -i corpus/base64 -o out/base64 -- build/native/fuzzer_base64 @@
#
# AFL++ will automatically dlopen the custom mutator via
# AFL_CUSTOM_MUTATOR_LIBRARY.  The mutator will try to connect to the mock
# daemon socket at /tmp/afl_ai_bridge.sock; if the daemon is not running it
# falls back to a local random operator (resilience policy).
# =============================================================================

# Resolve the directory containing this script (works when sourced or executed)
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PROJECT_DIR="$(dirname "$_SCRIPT_DIR")"   # adaptive-fuzzer/

# ---------------------------------------------------------------------------
# Required: point AFL++ at our custom mutator shared library
# ---------------------------------------------------------------------------
export AFL_CUSTOM_MUTATOR_LIBRARY="${_PROJECT_DIR}/c_mutator/libmutator.so"

# Use ONLY the custom mutator (disable AFL++ built-in havoc/splice).
# Set to 0 to run the custom mutator in addition to built-in stages.
export AFL_CUSTOM_MUTATOR_ONLY=1

# ---------------------------------------------------------------------------
# Optional AFL++ tuning flags
# ---------------------------------------------------------------------------
# Suppress the AFL++ banner (cleaner logs when running automated tests)
export AFL_NO_UI=1

# Suppress non-essential AFL++ status messages on stderr
export AFL_QUIET=1

# Bypass root checks for core_pattern and CPU governor
export AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
export AFL_SKIP_CPUFREQ=1

# IPC socket path (must match DEFAULT_SOCKET_PATH in protocol.h)
# Named without AFL_ prefix so AFL++ doesn't flag it as an unknown setting
export AI_BRIDGE_SOCK="/tmp/afl_ai_bridge.sock"

echo "[env.sh] AFL_CUSTOM_MUTATOR_LIBRARY=${AFL_CUSTOM_MUTATOR_LIBRARY}"
echo "[env.sh] Environment ready. Start mock daemon with:"
echo "         python3 ${_SCRIPT_DIR}/run_mock_daemon.py &"
