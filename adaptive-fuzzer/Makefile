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
