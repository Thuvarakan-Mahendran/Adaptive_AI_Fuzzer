# prompt.md

### Task: Audit Existing Harness & Header, and Refactor Zephyr `modbus_serial.c` into Standalone `modbus_rtu.c`

#### Context & Objectives
We are isolating the Modbus RTU serial frame handling logic from the upstream Zephyr RTOS tree (`subsys/modbus/modbus_serial.c`) to build a deterministic, standalone C fuzzing target (`targets/modbus/modbus_rtu.c`). 

We already have preliminary versions of:
- `targets/modbus/include/zephyr_compat.h`
- `targets/modbus/harness_modbus.c`

Before making destructive modifications to the upstream source or our existing support files, you must perform a static audit of both existing support files against the upstream implementation to ensure all naming conventions, signatures, types, and harness contracts align. Then, implement the six defined modifications to decouple the module from the RTOS and enable standalone compilation under Clang, and cross-compilers (`arm-linux-gnueabihf-gcc`).
sample clang command for native compilation:
AFL_SKIP_CPUFREQ=1 afl-fuzz -i corpus/modbus_seeds -o fuzzing_out/test_modbus -- ./build/native/fuzzer_modbus_native

Task 1: Static Pre-Audit of zephyr_compat.h and harness_modbus.cInspect subsys/modbus/modbus_serial.c alongside targets/modbus/include/zephyr_compat.h and targets/modbus/harness_modbus.c. Analyze the code to answer the following:API & Symbol Alignment in harness_modbus.c:What is the exact function name and signature that harness_modbus.c calls via extern (e.g., modbus_rtu_raw_rx, modbus_rx_packet_handler, modbus_raw_rx)?Does this function exist in modbus_serial.c, or is there a naming mismatch?How does the harness deliver data: byte-by-byte (serial loop) or as a full slice (const uint8_t *buf, size_t len)?Does the harness invoke an initialization/reset function between iterations to clear parser states and contexts?Completeness of zephyr_compat.h:Check if all required Zephyr data types, return error codes (EINVAL, ENOMEM, ETIMEDOUT, EIO), and data structures (such as struct modbus_context, struct modbus_iface_param, or device stubs) are declared with matching member layouts.Verify whether missing logging macros (LOG_ERR, LOG_WRN, LOG_INF, LOG_DBG) or synchronization primitives (k_sem, k_mutex) will trigger undeclared identifier warnings or compilation errors.Produce an Audit Assessment Report:Clearly state whether zephyr_compat.h and harness_modbus.c are READY TO GO or if specific signature/definition adjustments are required before proceeding.Task 2: Six Required Modifications on modbus_serial.c $\to$ modbus_rtu.cRefactor the copied upstream modbus_serial.c into targets/modbus/modbus_rtu.c by applying the following six modifications:   1. Header Isolation & Dependency DecouplingRemove or comment out all native Zephyr kernel, driver, and tree headers:C/* #include  */
/* #include  */
/* #include  */
/* #include  */
/* #include  */
/* #include  */
Replace them with standard C library headers (, , , , ``) and our local compatibility header:C#include "zephyr_compat.h"
Remove all device tree macros (DT_*), initialization macros (DEVICE_DEFINE, SYS_INIT), and runtime device binding queries.2. CRC Verification BypassLocate the 16-bit CRC calculation and check function in modbus_serial.c (e.g., modbus_rtu_check_crc or the conditional block evaluating frame checksum integrity).Modify this function or block so that it unconditionally evaluates to true (skipping the discard branch) to allow mutated, malformed inputs to explore deeper protocol states:   Cstatic inline bool modbus_rtu_check_crc(uint8_t *frame, uint16_t len) {
    return true; /* Checksum check bypassed per research paper */
}
3. Timers Replaced with Polling / Synchronous TransitionsLocate all references to k_timer (e.g., k_timer_init, k_timer_start, k_timer_stop, k_timer_status_get) governing inter-character ($t_{1.5}$) and inter-frame ($t_{3.5}$) silence delimiters.Remove all kernel timer dependencies.   Replace the asynchronous timeout logic with deterministic state transitions: once the input bytes provided by the harness are ingested, synchronously trigger the frame-end state transition (e.g., moving directly to MODBUS_STATE_FRAME_RCVD or dispatching the frame handler).   4. Frame Boundary Clamping (256-Byte Bound)Define and enforce the strict frame limit:   C#define MODBUS_MAX_FRAME_SIZE 256
Ensure the internal frame reception buffer cannot exceed 256 bytes.   Guard entry ingestion loops so inputs greater than 256 bytes are clamped or discarded before triggering unintended buffer overflows not present in bounded embedded deployments.   5. RTOS Synchronization & Logging RedirectionRemove or stub out RTOS synchronization primitives (k_sem_take, k_sem_give, k_mutex_lock, k_mutex_unlock) with no-op operations.Map error and diagnostic logging macros (LOG_ERR, LOG_WRN, LOG_INF) to standard I/O for triage[cite: 1, 2]:C#define LOG_ERR(...) printf("[LOG_ERR] " __VA_ARGS__)
#define LOG_WRN(...) printf("[LOG_WRN] " __VA_ARGS__)
#define LOG_INF(...) printf("[LOG_INF] " __VA_ARGS__)
#define LOG_DBG(...) /* no-op to optimize execution throughput */
6. Export Entry Point for Harness IntegrationExpose a clean, callable C entry point matching the audited contract in harness_modbus.c:Cvoid modbus_rtu_fuzz_inject(const uint8_t *data, size_t len);
Within this function:Reset the internal state machine and buffer pointers to their initial state.Feed the provided byte array sequentially through the Modbus RTU ingestion routines.Synchronously signal frame completion and trigger the protocol parser[cite: 1, 2].Task 3: Verification and Build ValidationAudit Alignment Summary:Detail any modifications made to zephyr_compat.h or harness_modbus.c to reconcile them with modbus_rtu.c
