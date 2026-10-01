# Adaptive Fuzzer Project Setup Guide

This document provides step-by-step instructions to set up the `adaptive-fuzzer/` project root with the modular architecture described in the paper. Follow each section to create the directory structure, implement the Modbus RTU and UART driver targets, and configure the build system.

---

## 1. Folder Structure Setup

Initialize the project root and create the specific targets and header paths matching the paper's modular architecture:

1. Initialize the root directory `adaptive-fuzzer/` and enter it:
   ```bash
   mkdir -p adaptive-fuzzer && cd adaptive-fuzzer
   ```

2. Under `adaptive-fuzzer/`, create the `targets/` directory:
   ```bash
   mkdir -p targets
   ```

3. Inside `targets/`, create two distinct subdirectories: `uart/` and `modbus/`:
   ```bash
   mkdir -p targets/uart targets/modbus
   ```

4. Inside `targets/uart/`, create an `include/` directory to store mock headers and definitions:
   ```bash
   mkdir -p targets/uart/include
   ```

5. Inside `targets/modbus/`, create an `include/` directory for protocol and mock header files:
   ```bash
   mkdir -p targets/modbus/include
   ```

6. Create empty source files for the implementation:
   ```bash
   touch targets/uart/uart_async_rx.c
   touch targets/uart/harness_uart.c
   touch targets/modbus/modbus_rtu.c
   touch targets/modbus/harness_modbus.c
   ```

7. Create `corpus/uart_seeds/` and `corpus/modbus_seeds/` for initial input storage:
   ```bash
   mkdir -p corpus/uart_seeds corpus/modbus_seeds
   ```

8. Create `build/native/` and `build/arm/` to hold compiled artifacts:
   ```bash
   mkdir -p build/native build/arm
   ```

---

## 2. Modbus RTU Target Implementation (`targets/modbus/`)

Follow these steps to extract and adapt the Zephyr Modbus logic as described in Section III-C of the paper.

### Step 2.1: Locate and Extract Source Files

1. Fetch the official Zephyr RTOS repository:
   ```bash
   git clone https://github.com/zephyrproject-rtos/zephyr.git
   ```

2. Locate the Modbus RTU implementation files:
   - Primary source: `subsys/modbus/modbus_rtu.c`
   - Associated header: `include/zephyr/modbus/modbus.h`

3. Copy the parsing and state machine functions into `targets/modbus/modbus_rtu.c`.

### Step 2.2: Mock Zephyr RTOS Dependencies

In `targets/modbus/include/zephyr_compat.h`:

1. Define basic standard integer types (`uint8_t`, `uint16_t`, `uint32_t`, `bool`) using `<stdint.h>` and `<stdbool.h>`.

2. Replace Zephyr's logging macros (`LOG_ERR`, `LOG_WRN`, `LOG_INF`, `LOG_DBG`) with standard library print calls (`printf(...)` or conditional debug stubs) as specified by the paper.

3. Stub out RTOS kernel sync objects (such as `k_sem`, `k_mutex`) with standard no-op structures so the parser does not block.

### Step 2.3: Strip CRC Validation

1. Identify the frame integrity function in `modbus_rtu.c` (typically a 16-bit CRC check function like `modbus_rtu_check_crc()`).

2. Modify this function to bypass verification: force it to always return `true` or bypass the branch that drops invalid CRC packets.

   **Reason:** Fuzzers struggle to generate valid cyclic redundancy checks randomly; bypassing it allows the fuzzer to directly test deeper command-handling states without early rejections.

### Step 2.4: Replace Timers with Polling Transitions

1. Locate RTOS timer callbacks that monitor inter-frame and inter-character silent intervals (e.g., `t1_5` and `t3_5` silence detection timers).

2. Remove hardware timer calls (e.g., `k_timer_start()`, `k_timer_stop()`).

3. Convert timeout-driven state changes into synchronous transitions that evaluate immediately upon receiving an end-of-frame indicator or when the input buffer is exhausted.

### Step 2.5: Enforce Frame Capping

1. Define a preprocessor boundary macro `MODBUS_MAX_FRAME_SIZE` set strictly to `256`.

2. Add an input length check at the entry of the frame processing logic to reject any buffer exceeding 256 bytes.

### Step 2.6: Implement the Modbus Harness (`harness_modbus.c`)

1. Include standard headers (`<unistd.h>`, `<stdint.h>`, `<assert.h>`).

2. Include the mocked headers and target function declarations.

3. Define the entry-point function (e.g., `void fuzz_target_entry(void)`) to serve as the persistent hook anchor:
   - Allocate a local static buffer of size 256 bytes.
   - Read raw bytes from standard input (`STDIN_FILENO`) up to the buffer size.
   - Add a state-reset routine to reset Modbus state machine variables (e.g., slave address, transaction ID, internal state) between iterations.
   - Call the extracted Modbus frame handler passing the buffer and the read byte count.
   - Add post-execution assertions to detect abnormal control-flow states or illegal pointer jumps (triggering `SIGABRT` upon violation).

4. Implement `main(void)` to invoke `fuzz_target_entry()` for single-run or persistent loop invocation.

---

## 3. UART Driver Target Implementation (`targets/uart/`)

Follow these steps to extract and adapt the Zephyr asynchronous UART driver logic.

### Step 3.1: Locate and Extract Source Files

1. From the Zephyr repository, extract the asynchronous reception state handling logic (often located in `drivers/serial/` or asynchronous driver helper libraries like `uart_async_rx.c`).

2. Place the logic in `targets/uart/uart_async_rx.c`.

### Step 3.2: Mock Zephyr RTOS Dependencies

In `targets/uart/include/zephyr_compat.h`:

1. Provide standard type definitions (`uint8_t`, `size_t`, `int32_t`).

2. Redirect error logging macros (`LOG_ERR`) to `printf()`.

3. Stub out system device trees (`DEVICE_DT_GET`) and device binding structures with a single mock hardware structure.

### Step 3.3: Replace ISR Handling with a Simulated FIFO

1. Remove all bare-metal interrupt handlers, hardware interrupt registration (`IRQ_CONNECT`), and interrupt enable/disable primitives (`irq_enable()`, `irq_disable()`).

2. Implement a software ring buffer / FIFO data structure in C inside `uart_async_rx.c`.

3. Replace the driver's interrupt-driven read routine so that instead of querying a hardware UART RX register, it pulls bytes sequentially from this internal software FIFO.

### Step 3.4: Replace DMA Interactions with Static Memory Pools

1. Locate Direct Memory Access (DMA) channel configuration routines and buffer-swapping functions.

2. Remove all peripheral DMA control registers and hardware-driven scatter-gather descriptors.

3. Define a static memory pool (e.g., a two-dimensional fixed-size array) to act as double-buffered reception targets.

4. Modify the buffer exchange routine to copy data directly from the simulated FIFO into these pre-allocated static pools without triggering hardware bus requests.

### Step 3.5: Replace Polling Loops

1. Identify any busy-wait polling loops checking hardware status flags (e.g., checking if the RX FIFO is non-empty).

2. Replace these loops with counter-bounded checks against the simulated software FIFO to prevent the fuzzer from hanging indefinitely.

### Step 3.6: Implement the UART Harness (`harness_uart.c`)

1. Include system I/O headers and the local mocked UART interface.

2. Define the fuzz entry function (e.g., `void fuzz_uart_entry(void)`):
   - Allocate a local buffer to ingest mutated test cases.
   - Read test data from `STDIN_FILENO`.
   - Reset driver state: clear the static memory pools, zero out read/write pointers of the simulated FIFO, and reset the receiver state machine.
   - Push the fuzzer-provided input bytes directly into the simulated FIFO to mimic incoming peripheral stream bytes.
   - Trigger the driver's receive processing routine so it transfers data from the FIFO to the static pool.
   - Add boundary assertions verifying that the static reception pools did not suffer an overflow (checking for off-by-one or buffer-boundary overwrites).

3. Implement `main(void)` to call the entry function.

---

## 4. Build Configuration

Create a root `Makefile` inside `adaptive-fuzzer/` to manage compilation workflows across tiers.

### Rule 1: Native Compilation (Tier A)

- Set compiler to `afl-clang-fast`.
- Include compile flags for AddressSanitizer (`-fsanitize=address`) and optimization (`-O2`).
- Compile `targets/modbus/` source and harness into `build/native/fuzzer_modbus_native`.
- Compile `targets/uart/` source and harness into `build/native/fuzzer_uart_native`.

### Rule 2: Cross-Architecture Compilation (Tier B)

- Set cross-compilers:
  - `arm-linux-gnueabihf-gcc` (for ARM Cortex-M emulation target)
  - `riscv64-linux-gnu-gcc` (for RISC-V emulation target)
- Add the `-static` flag to ensure binaries bundle standard C dependencies and can run cleanly inside `qemu-arm` and `qemu-riscv64`.
- Compile `targets/modbus/` source and harness into `build/arm/fuzzer_modbus_arm`.
- Compile `targets/uart/` source and harness into `build/arm/fuzzer_uart_arm`.