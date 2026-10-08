# =============================================================================
# ipc_bridge/Makefile – Standalone build harness for IPC client & test suite
# =============================================================================

CC      := clang-14
CFLAGS  := -O3 -Wall -Wextra -Icommon -Iclient_c
FPIC    := -fPIC

CLIENT_OBJ := client_c/socket_client.o
TEST_BIN   := tests/test_ipc_client

.PHONY: all tests clean

all: $(CLIENT_OBJ) $(TEST_BIN)

$(CLIENT_OBJ): client_c/socket_client.c client_c/socket_client.h common/protocol.h
	$(CC) $(CFLAGS) $(FPIC) -c $< -o $@
	@echo "[ipc_bridge] Compiled $@"

$(TEST_BIN): tests/test_ipc_client.c $(CLIENT_OBJ) common/protocol.h client_c/socket_client.h
	$(CC) $(CFLAGS) $< $(CLIENT_OBJ) -o $@
	@echo "[ipc_bridge] Built $@"

tests: all
	@./tests/run_ipc_check.sh

clean:
	rm -f $(CLIENT_OBJ) $(TEST_BIN)
	@echo "[ipc_bridge] Cleaned build artefacts"
