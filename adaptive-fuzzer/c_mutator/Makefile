# =============================================================================
# c_mutator/Makefile – AFL++ custom mutator shared library + unit test binary
#
# Standalone Makefile (not merged into the top-level adaptive-fuzzer/Makefile).
#
# Targets:
#   all           Build libmutator.so and test_agg (default)
#   libmutator.so AFL++ custom mutator shared library
#   test_agg      Aggregator unit test binary (does not link against libmutator)
#   clean         Remove all build artefacts
#
# Compiler: clang-14 (matches the LLVM-14 toolchain used in the rest of the
#           project).  libmutator.so MUST NOT be compiled with afl-clang-fast
#           because the shared library is loaded by AFL++ itself, not instrumented.
#
# AFL++ headers: pinned to ../../AFLplusplus/include (relative to this file).
#   Key headers pulled in transitively by mutator.h → afl-fuzz.h:
#     types.h, config.h, forkserver.h, alloc-inl.h, debug.h, hash.h
# =============================================================================

CC      := clang-14
CFLAGS  := -O3 -fPIC -Wall -Wextra -I../../AFLplusplus/include
LDFLAGS := -shared -lm

# Source files for the shared library
LIB_SRCS := aggregator.c operators.c mutator.c
LIB_OBJS := $(LIB_SRCS:.c=.o)

# Default target
.PHONY: all
all: libmutator.so test_agg

# -----------------------------------------------------------------------------
# Shared library
# -----------------------------------------------------------------------------
libmutator.so: $(LIB_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^
	@echo "[c_mutator] Built $@ ($(shell wc -c < $@) bytes)"

# Object files – explicit prerequisites to track header changes
aggregator.o: aggregator.c aggregator.h protocol.h
	$(CC) $(CFLAGS) -c $< -o $@

operators.o: operators.c operators.h
	$(CC) $(CFLAGS) -c $< -o $@

mutator.o: mutator.c mutator.h protocol.h aggregator.h operators.h
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
