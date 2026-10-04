# =============================================================================
# Makefile – adaptive-fuzzer build system
#
# Manages two compilation tiers:
#   Tier A (native):  AFL++-instrumented binaries with AddressSanitizer
#   Tier B (cross):   Static ARM / RISC-V binaries for QEMU emulation
#
# Usage:
#   make native         – build both native fuzzer binaries
#   make arm            – cross-compile both ARM static binaries
#   make riscv          – cross-compile both RISC-V static binaries
#   make all            – build all tiers
#   make clean          – remove all build artefacts
#   make run_modbus     – quick single-input smoke test (Modbus, native)
#   make run_uart       – quick single-input smoke test (UART, native)
# =============================================================================

# -----------------------------------------------------------------------------
# Directories
# -----------------------------------------------------------------------------

BUILD_NATIVE   := build/native
BUILD_ARM      := build/arm
BUILD_RISCV    := build/riscv

MODBUS_SRCS    := targets/modbus/modbus_rtu.c targets/modbus/harness_modbus.c
UART_SRCS      := targets/uart/uart_async_rx.c targets/uart/harness_uart.c

MODBUS_INC     := -Itargets/modbus/include
UART_INC       := -Itargets/uart/include

# -----------------------------------------------------------------------------
# Tier A: Native compilation (Rule 1)
#
#   Compiler : afl-clang-fast  (AFL++ coverage-tracking instrumentation)
#   Sanitizer: AddressSanitizer (-fsanitize=address) to catch memory errors
#   Opt level: -O2
# -----------------------------------------------------------------------------

AFL_WRAPPER := afl-clang-fast
REAL_CLANG  := /usr/lib/llvm-14/bin/clang
NATIVE_CFLAGS  := -O2 -fsanitize=address -g -Wall -Wextra \
                  -Wno-unused-parameter -Wno-unused-function

MODBUS_NATIVE  := $(BUILD_NATIVE)/fuzzer_modbus_native
UART_NATIVE    := $(BUILD_NATIVE)/fuzzer_uart_native

.PHONY: native native_modbus native_uart

native: native_modbus native_uart

native_modbus: $(MODBUS_NATIVE)
$(MODBUS_NATIVE): $(MODBUS_SRCS) | $(BUILD_NATIVE)
	AFL_CC=$(REAL_CLANG) $(AFL_WRAPPER) --afl-llvm $(NATIVE_CFLAGS) $(MODBUS_INC) $(MODBUS_SRCS) -o $@
	@echo "[Tier A] Built $@"

native_uart: $(UART_NATIVE)
$(UART_NATIVE): $(UART_SRCS) | $(BUILD_NATIVE)
	AFL_CC=$(REAL_CLANG) $(AFL_WRAPPER) --afl-llvm $(NATIVE_CFLAGS) $(UART_INC) $(UART_SRCS) -o $@
	@echo "[Tier A] Built $@"

# -----------------------------------------------------------------------------
# Tier B: Cross-architecture compilation (Rule 2)
#
#   ARM:    arm-linux-gnueabihf-gcc   → qemu-arm
#   RISC-V: riscv64-linux-gnu-gcc    → qemu-riscv64
#
#   Flag: -static  – bundles libc so binaries run inside baremetal QEMU
# -----------------------------------------------------------------------------

ARM_CC         := arm-linux-gnueabihf-gcc
RISCV_CC       := riscv64-linux-gnu-gcc
CROSS_CFLAGS   := -O2 -static -g -Wall -Wextra \
                  -Wno-unused-parameter -Wno-unused-function

MODBUS_ARM     := $(BUILD_ARM)/fuzzer_modbus_arm
UART_ARM       := $(BUILD_ARM)/fuzzer_uart_arm

MODBUS_RISCV   := $(BUILD_RISCV)/fuzzer_modbus_riscv
UART_RISCV     := $(BUILD_RISCV)/fuzzer_uart_riscv

.PHONY: arm arm_modbus arm_uart riscv riscv_modbus riscv_uart

arm: arm_modbus arm_uart

arm_modbus: $(MODBUS_ARM)
$(MODBUS_ARM): $(MODBUS_SRCS) | $(BUILD_ARM)
	$(ARM_CC) $(CROSS_CFLAGS) $(MODBUS_INC) $(MODBUS_SRCS) -o $@
	@echo "[Tier B / ARM] Built $@"

arm_uart: $(UART_ARM)
$(UART_ARM): $(UART_SRCS) | $(BUILD_ARM)
	$(ARM_CC) $(CROSS_CFLAGS) $(UART_INC) $(UART_SRCS) -o $@
	@echo "[Tier B / ARM] Built $@"

riscv: riscv_modbus riscv_uart

riscv_modbus: $(MODBUS_RISCV)
$(MODBUS_RISCV): $(MODBUS_SRCS) | $(BUILD_RISCV)
	$(RISCV_CC) $(CROSS_CFLAGS) $(MODBUS_INC) $(MODBUS_SRCS) -o $@
	@echo "[Tier B / RISC-V] Built $@"

riscv_uart: $(UART_RISCV)
$(UART_RISCV): $(UART_SRCS) | $(BUILD_RISCV)
	$(RISCV_CC) $(CROSS_CFLAGS) $(UART_INC) $(UART_SRCS) -o $@
	@echo "[Tier B / RISC-V] Built $@"

# -----------------------------------------------------------------------------
# all: build every tier
# -----------------------------------------------------------------------------

.PHONY: all
all: native arm riscv
	@echo "[all] All tiers built."

# -----------------------------------------------------------------------------
# Smoke tests (single-run with /dev/null as the corpus seed)
# -----------------------------------------------------------------------------

.PHONY: run_modbus run_uart

run_modbus: $(MODBUS_NATIVE)
	@echo "[smoke] Running Modbus harness with empty input …"
	echo -n "" | $(MODBUS_NATIVE) && echo "PASS" || echo "FAIL (non-zero exit)"

run_uart: $(UART_NATIVE)
	@echo "[smoke] Running UART harness with empty input …"
	echo -n "" | $(UART_NATIVE) && echo "PASS" || echo "FAIL (non-zero exit)"

# -----------------------------------------------------------------------------
# Seed corpus helpers
# -----------------------------------------------------------------------------

.PHONY: seeds

seeds:
	# Minimal valid Modbus RTU frames as initial corpus seeds
	# Frame: addr=0x01 fc=0x03 start=0x0000 qty=0x0001 crc=0x840A (CRC bypassed)
	printf '\x01\x03\x00\x00\x00\x01\x84\x0A' > corpus/modbus_seeds/read_holding_reg.bin
	# Frame: addr=0x01 fc=0x06 reg=0x0001 val=0x0003 crc=0x9869 (CRC bypassed)
	printf '\x01\x06\x00\x01\x00\x03\x98\x69' > corpus/modbus_seeds/write_single_reg.bin
	# Minimal UART seed: 4 bytes of incrementing data
	printf '\x01\x02\x03\x04' > corpus/uart_seeds/simple_stream.bin
	@echo "[seeds] Initial corpus seeds created."

# -----------------------------------------------------------------------------
# Directory guards
# -----------------------------------------------------------------------------

$(BUILD_NATIVE) $(BUILD_ARM) $(BUILD_RISCV):
	mkdir -p $@

# -----------------------------------------------------------------------------
# Clean
# -----------------------------------------------------------------------------

.PHONY: clean
clean:
	rm -rf $(BUILD_NATIVE) $(BUILD_ARM) $(BUILD_RISCV)
	@echo "[clean] Build artefacts removed."

# =============================================================================
# LAVA-M Coreutils targets
#
# These are full Coreutils 8.24 source trees instrumented with AFL++.
# They CANNOT be built with a direct $(AFL_CC) $(SRCS) -o $@ rule because:
#   1. Each tree needs generated config headers (config.h, lib/config.h)
#      produced by ./configure; without them ~200 Gnulib source files fail.
#   2. libcoreutils.a and the Gnulib portability library (libgnu.a) are
#      intermediate archives linked into each utility; these must be built
#      by the recursive Autotools make, not hand-assembled.
#   3. Autotools-generated sources (version.c, etc.) do not exist before
#      make runs; the dependency graph is managed by the generated Makefiles.
#   4. The existing NATIVE_CFLAGS include -fsanitize=address which conflicts
#      with the Autotools configure self-test programs; LAVA-M builds pass
#      their own CFLAGS='-g -O2' through the sub-make.
#
# Design: each target delegates entirely to the Autotools sub-build, then
# copies the finished binary into $(BUILD_NATIVE) under the fuzzer_* name.
# Source trees live in targets/lava_m/<util>/coreutils-8.24-lava-safe/ and
# are copies of the originals; the upstream corpus is never modified.
# =============================================================================

# -----------------------------------------------------------------------------
# LAVA-M source and output paths
# -----------------------------------------------------------------------------

LAVA_BASE   := targets/lava_m
LAVA_JOBS   := $(shell nproc)

#LAVA_CC     := afl-clang-fast
#LAVA_CXX    := afl-clang-fast++
# -DSLOW_BUT_NO_HACKS=1: gnulib freadseek.c portability fix.
# glibc 2.35+ (Ubuntu 22.04+) removed _IO_ftrylockfile which gnulib used to
# detect GNU libc and select a fast file-pointer path.  Without this flag,
# freadseek.c emits a #error and the build fails.  SLOW_BUT_NO_HACKS forces
# the generic fseek-based path, which is fully correct for fuzzing purposes.
#LAVA_CFLAGS := -g -O2 -DSLOW_BUT_NO_HACKS=1
LAVA_WRAPPER := afl-clang-fast
LAVA_REAL_CC := /usr/lib/llvm-14/bin/clang
LAVA_REAL_CXX := /usr/lib/llvm-14/bin/clang++
LAVA_CFLAGS := -g -O2 -DSLOW_BUT_NO_HACKS=1 \
               -include $(CURDIR)/targets/lava_m/lava_compat.h

FUZZER_BASE64 := $(BUILD_NATIVE)/fuzzer_base64
FUZZER_MD5SUM := $(BUILD_NATIVE)/fuzzer_md5sum
FUZZER_UNIQ   := $(BUILD_NATIVE)/fuzzer_uniq
FUZZER_WHO    := $(BUILD_NATIVE)/fuzzer_who

# Corpus directories
CORPUS_BASE64 := corpus/lava_base64
CORPUS_MD5SUM := corpus/lava_md5sum
CORPUS_UNIQ   := corpus/lava_uniq
CORPUS_WHO    := corpus/lava_who

.PHONY: lava lava_base64 lava_md5sum lava_uniq lava_who clean_lava smoke_lava

# Build all four LAVA-M utilities
lava: lava_base64 lava_md5sum lava_uniq lava_who
	@echo "[LAVA-M] All four targets staged under $(BUILD_NATIVE)."

# ---- base64 -----------------------------------------------------------------
lava_base64: $(FUZZER_BASE64)

$(FUZZER_BASE64): | $(BUILD_NATIVE) $(CORPUS_BASE64)
	@echo "[LAVA-M] Configuring base64 ..."
	cd $(LAVA_BASE)/base64/coreutils-8.24-lava-safe && \
	  FORCE_UNSAFE_CONFIGURE=1 \
	  CC='env AFL_CC=$(LAVA_REAL_CC) $(LAVA_WRAPPER) --afl-llvm' \
	  CXX='env AFL_CC=$(LAVA_REAL_CXX) $(LAVA_WRAPPER) --afl-llvm' \
	  CFLAGS='$(LAVA_CFLAGS)' \
	  ./configure --prefix="$$(pwd)/install" --disable-nls --disable-man
	@echo "[LAVA-M] Building base64 ..."
	$(MAKE) -C $(LAVA_BASE)/base64/coreutils-8.24-lava-safe -j$(LAVA_JOBS) \
	  HELP2MAN=true \
	  man1_MANS= \
	  EXTRA_MANS=
	@test -f $(LAVA_BASE)/base64/coreutils-8.24-lava-safe/src/base64 || \
	  (echo "[LAVA-M] ERROR: src/base64 not found after build" && exit 1)
	cp $(LAVA_BASE)/base64/coreutils-8.24-lava-safe/src/base64 $@
	@echo "[LAVA-M] Staged $@"

# ---- md5sum -----------------------------------------------------------------
lava_md5sum: $(FUZZER_MD5SUM)

$(FUZZER_MD5SUM): | $(BUILD_NATIVE) $(CORPUS_MD5SUM)
	@echo "[LAVA-M] Configuring md5sum ..."
	cd $(LAVA_BASE)/md5sum/coreutils-8.24-lava-safe && \
	  FORCE_UNSAFE_CONFIGURE=1 \
          CC='env AFL_CC=$(LAVA_REAL_CC) $(LAVA_WRAPPER) --afl-llvm' \
          CXX='env AFL_CC=$(LAVA_REAL_CXX) $(LAVA_WRAPPER) --afl-llvm' \
          CFLAGS='$(LAVA_CFLAGS)' \
          ./configure --prefix="$$(pwd)/install" --disable-nls --disable-man
	@echo "[LAVA-M] Building md5sum ..."
	$(MAKE) -C $(LAVA_BASE)/md5sum/coreutils-8.24-lava-safe -j$(LAVA_JOBS) \
          HELP2MAN=true \
          man1_MANS= \
          EXTRA_MANS=
	@test -f $(LAVA_BASE)/md5sum/coreutils-8.24-lava-safe/src/md5sum || \
	  (echo "[LAVA-M] ERROR: src/md5sum not found after build" && exit 1)
	cp $(LAVA_BASE)/md5sum/coreutils-8.24-lava-safe/src/md5sum $@
	@echo "[LAVA-M] Staged $@"

# ---- uniq -------------------------------------------------------------------
lava_uniq: $(FUZZER_UNIQ)

$(FUZZER_UNIQ): | $(BUILD_NATIVE) $(CORPUS_UNIQ)
	@echo "[LAVA-M] Configuring uniq ..."
	cd $(LAVA_BASE)/uniq/coreutils-8.24-lava-safe && \
	  FORCE_UNSAFE_CONFIGURE=1 \
          CC='env AFL_CC=$(LAVA_REAL_CC) $(LAVA_WRAPPER) --afl-llvm' \
          CXX='env AFL_CC=$(LAVA_REAL_CXX) $(LAVA_WRAPPER) --afl-llvm' \
          CFLAGS='$(LAVA_CFLAGS)' \
          ./configure --prefix="$$(pwd)/install" --disable-nls --disable-man
	@echo "[LAVA-M] Building uniq ..."
	$(MAKE) -C $(LAVA_BASE)/uniq/coreutils-8.24-lava-safe -j$(LAVA_JOBS) \
          HELP2MAN=true \
          man1_MANS= \
          EXTRA_MANS=
	@test -f $(LAVA_BASE)/uniq/coreutils-8.24-lava-safe/src/uniq || \
	  (echo "[LAVA-M] ERROR: src/uniq not found after build" && exit 1)
	cp $(LAVA_BASE)/uniq/coreutils-8.24-lava-safe/src/uniq $@
	@echo "[LAVA-M] Staged $@"

# ---- who --------------------------------------------------------------------
lava_who: $(FUZZER_WHO)

$(FUZZER_WHO): | $(BUILD_NATIVE) $(CORPUS_WHO)
	@echo "[LAVA-M] Configuring who ..."
	cd $(LAVA_BASE)/who/coreutils-8.24-lava-safe && \
	  FORCE_UNSAFE_CONFIGURE=1 \
          CC='env AFL_CC=$(LAVA_REAL_CC) $(LAVA_WRAPPER) --afl-llvm' \
          CXX='env AFL_CC=$(LAVA_REAL_CXX) $(LAVA_WRAPPER) --afl-llvm' \
          CFLAGS='$(LAVA_CFLAGS)' \
          ./configure --prefix="$$(pwd)/install" --disable-nls --disable-man
	@echo "[LAVA-M] Building who ..."
	$(MAKE) -C $(LAVA_BASE)/who/coreutils-8.24-lava-safe -j$(LAVA_JOBS) \
          HELP2MAN=true \
          man1_MANS= \
          EXTRA_MANS=
	@test -f $(LAVA_BASE)/who/coreutils-8.24-lava-safe/src/who || \
	  (echo "[LAVA-M] ERROR: src/who not found after build" && exit 1)
	cp $(LAVA_BASE)/who/coreutils-8.24-lava-safe/src/who $@
	@echo "[LAVA-M] Staged $@"

# -----------------------------------------------------------------------------
# clean_lava – remove staged LAVA-M binaries only
# Does NOT delete source copies or the upstream corpus.
# -----------------------------------------------------------------------------

clean_lava:
	rm -f $(FUZZER_BASE64) $(FUZZER_MD5SUM) $(FUZZER_UNIQ) $(FUZZER_WHO)
	@echo "[LAVA-M] Staged binaries removed. Source trees preserved."

# -----------------------------------------------------------------------------
# smoke_lava – functional smoke tests for all four staged binaries
# Verifies staged binaries exist before running any test.
# -----------------------------------------------------------------------------

smoke_lava: $(FUZZER_BASE64) $(FUZZER_MD5SUM) $(FUZZER_UNIQ) $(FUZZER_WHO)
	@echo "[smoke_lava] Verifying staged binaries ..."
	@test -x $(FUZZER_BASE64) || (echo "MISSING: $(FUZZER_BASE64)" && exit 1)
	@test -x $(FUZZER_MD5SUM) || (echo "MISSING: $(FUZZER_MD5SUM)" && exit 1)
	@test -x $(FUZZER_UNIQ)   || (echo "MISSING: $(FUZZER_UNIQ)"   && exit 1)
	@test -x $(FUZZER_WHO)    || (echo "MISSING: $(FUZZER_WHO)"     && exit 1)
	@echo "--- base64 smoke test ---"
	echo "SGVsbG8gV29ybGQh" > $(CORPUS_BASE64)/seed.txt
	$(FUZZER_BASE64) -d $(CORPUS_BASE64)/seed.txt
	@echo "--- md5sum smoke test ---"
	printf 'hello world' > $(CORPUS_MD5SUM)/test.txt
	md5sum $(CORPUS_MD5SUM)/test.txt > $(CORPUS_MD5SUM)/seed.txt
	$(FUZZER_MD5SUM) -c $(CORPUS_MD5SUM)/seed.txt
	@echo "--- uniq smoke test ---"
	printf 'line1\nline1\nline2\n' > $(CORPUS_UNIQ)/seed.txt
	$(FUZZER_UNIQ) $(CORPUS_UNIQ)/seed.txt
	@echo "--- who smoke test ---"
	$(FUZZER_WHO) $(CORPUS_WHO)/utmp.clean 2>&1 || true
	@echo "[smoke_lava] All smoke tests completed."

# -----------------------------------------------------------------------------
# Corpus directory guards for LAVA-M
# -----------------------------------------------------------------------------

$(CORPUS_BASE64) $(CORPUS_MD5SUM) $(CORPUS_UNIQ) $(CORPUS_WHO):
	mkdir -p $@
