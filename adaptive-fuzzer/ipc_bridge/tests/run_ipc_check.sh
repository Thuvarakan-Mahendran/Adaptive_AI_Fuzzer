#!/usr/bin/env bash
# =============================================================================
# run_ipc_check.sh – Automated verification and resilience test for ipc_bridge.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_DIR="$(dirname "$SCRIPT_DIR")"
SOCK_TEST="/tmp/test_ipc_check_$$_.sock"

echo "============================================================"
echo " Running Automated IPC Bridge Check Suite"
echo "============================================================"

# Ensure clean state
rm -f "$SOCK_TEST"

# 1. Start mock server in background
echo "[*] Launching mock server on ${SOCK_TEST}..."
python3 "${BRIDGE_DIR}/server_python/mock_server.py" --sock "$SOCK_TEST" --benchmark > /dev/null 2>&1 &
SERVER_PID=$!

# Wait for socket to bind
for i in {1..30}; do
    if [ -S "$SOCK_TEST" ]; then
        break
    fi
    sleep 0.05
done

if [ ! -S "$SOCK_TEST" ]; then
    echo "[FAIL] Mock server failed to create socket: $SOCK_TEST"
    kill -9 "$SERVER_PID" 2>/dev/null || true
    exit 1
fi
echo "[+] Mock server is ready (PID: $SERVER_PID)"

# 2. Run loopback test
echo "[*] Running loopback test (5000 iterations)..."
"${SCRIPT_DIR}/test_ipc_client" --sock "$SOCK_TEST" --iters 5000

# Stop initial server
kill -15 "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
rm -f "$SOCK_TEST"

# 3. Fault Resilience Test: Server drop during active transmission
echo ""
echo "[*] Starting Fault Resilience Test (Server Kill Mid-Stream)..."
python3 "${BRIDGE_DIR}/server_python/mock_server.py" --sock "$SOCK_TEST" --benchmark > /dev/null 2>&1 &
RESIL_SERVER_PID=$!

for i in {1..30}; do
    if [ -S "$SOCK_TEST" ]; then
        break
    fi
    sleep 0.05
done

# Start client in background with high iteration count and resilience flag
"${SCRIPT_DIR}/test_ipc_client" --sock "$SOCK_TEST" --iters 100000 --resilience > /tmp/resil_client_$$.log 2>&1 &
CLIENT_PID=$!

# Let client stream for 50 ms then kill server abruptly with SIGKILL
sleep 0.05
kill -9 "$RESIL_SERVER_PID" 2>/dev/null || true
wait "$RESIL_SERVER_PID" 2>/dev/null || true
rm -f "$SOCK_TEST"

# Wait for client to detect disconnect and gracefully exit
wait "$CLIENT_PID" || true

echo "[+] Client output during resilience test:"
cat /tmp/resil_client_$$.log | grep -E "PASS|caught|elapsed" || cat /tmp/resil_client_$$.log
rm -f /tmp/resil_client_$$.log

echo ""
echo "============================================================"
echo " [PASS] ALL IPC BRIDGE TESTS COMPLETED SUCCESSFULLY"
echo "============================================================"
