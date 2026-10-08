# =============================================================================
# c_mutator/Makefile – AFL++ custom mutator shared library + unit test binary
#
# Standalone Makefile linking against decoupled ipc_bridge.
#
# Targets:
#   all           Build libmutator.so and test_agg (default)
#   libmutator.so AFL++ custom mutator shared library (links socket_client.o)
#   test_agg      Aggregator unit test binary (does not link against libmutator)
#   clean         Remove all build artefacts
#
# Compiler: clang-14 (matches LLVM-14 toolchain).
# =============================================================================

CC      := clang-14
IPC_DIR := ../ipc_bridge

CFLAGS  := -O3 -fPIC -Wall -Wextra \
           -I../../AFLplusplus/include \
           -I$(IPC_DIR)/common \
           -I$(IPC_DIR)/client_c

LDFLAGS := -shared -lm

# Source files for the shared library
LIB_SRCS := aggregator.c operators.c mutator.c
LIB_OBJS := $(LIB_SRCS:.c=.o)
IPC_OBJ  := $(IPC_DIR)/client_c/socket_client.o

# Default target
.PHONY: all
all: libmutator.so test_agg

# -----------------------------------------------------------------------------
# IPC Bridge Client Object Dependency
# -----------------------------------------------------------------------------
$(IPC_OBJ):
	$(MAKE) -C $(IPC_DIR) client_c/socket_client.o

# -----------------------------------------------------------------------------
# Shared library
# -----------------------------------------------------------------------------
libmutator.so: $(LIB_OBJS) $(IPC_OBJ)
	$(CC) $(LDFLAGS) -o $@ $^
	@echo "[c_mutator] Built $@ ($(shell wc -c < $@) bytes)"

# Object files – explicit prerequisites to track header changes
aggregator.o: aggregator.c aggregator.h protocol.h $(IPC_DIR)/common/protocol.h
	$(CC) $(CFLAGS) -c $< -o $@

operators.o: operators.c operators.h
	$(CC) $(CFLAGS) -c $< -o $@

mutator.o: mutator.c mutator.h protocol.h aggregator.h operators.h $(IPC_DIR)/client_c/socket_client.h
	$(CC) $(CFLAGS) -c $< -o $@

# -----------------------------------------------------------------------------
# Unit test binary (aggregator only – no AFL++ link-time deps)
# -----------------------------------------------------------------------------
test_agg: tests/test_aggregator.c aggregator.o
	$(CC) $(CFLAGS) tests/test_aggregator.c aggregator.o -o $@ -lm
	@echo "[c_mutator] Built $@"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
.PHONY: clean
clean:
	rm -f $(LIB_OBJS) libmutator.so test_agg
	@echo "[c_mutator] Clean done"
