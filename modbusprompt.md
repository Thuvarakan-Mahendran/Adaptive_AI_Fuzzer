# Task: Audit and Refactor Zephyr Modbus RTU for Standalone Fuzzing

## 1. Context and Objectives

We are isolating the Modbus RTU serial-frame handling logic from the upstream Zephyr RTOS implementation, `targets/modbus/modbus_serial.c`, to create a deterministic, standalone C fuzzing target:

```text
targets/modbus/modbus_rtu.c
```

The project already contains preliminary versions of:

- `targets/modbus/include/zephyr_compat.h`
- `targets/modbus/harness_modbus.c`

Before making destructive changes to the upstream source or existing support files, perform a static audit of both support files against the upstream implementation. Verify that all naming conventions, function signatures, types, data structures, and harness contracts are aligned.

Then refactor the copied implementation so that it is decoupled from Zephyr and can be compiled as a standalone target with Clang and cross-compilers such as `arm-linux-gnueabihf-gcc`.

Example native compilation command inside `adaptive-fuzzer/` folder:

```text
make -B native
```
Example arm compilation command inside `adaptive-fuzzer/` folder:

```text
make arm
```

## 2. Files to Inspect

Inspect the following files together:

1. `targets/modbus/modbus_serial.c`
2. `targets/modbus/include/zephyr_compat.h`
3. `targets/modbus/harness_modbus.c`

Use the upstream implementation as the reference for API names, signatures, structure layouts, state-machine behavior, and required dependencies.

## 3. Task 1: Static Pre-Audit

### 3.1 Harness API and Symbol Alignment

Analyze `harness_modbus.c` and answer the following:

- What exact function name and signature does the harness declare or call through `extern`?
- Does that function exist in `modbus_serial.c`?
- If not, identify the naming or signature mismatch.
- Does the harness deliver input byte-by-byte through a serial-style loop, or as a complete slice such as `const uint8_t *buf, size_t len`?
- Does the harness invoke an initialization or reset function between fuzzing iterations?
- Are parser state, frame buffers, counters, and protocol contexts cleared between iterations?
- Does the harness correctly handle zero-length input and inputs larger than the supported frame size?

### 3.2 Compatibility Header Completeness

Audit `zephyr_compat.h` against every symbol used by the refactored implementation. Verify that it declares or defines all required:

- Standard integer and Boolean types.
- Zephyr-compatible data types.
- Error codes, including `EINVAL`, `ENOMEM`, `ETIMEDOUT`, and `EIO`.
- Modbus-related structures, including, where required:
  - `struct modbus_context`
  - `struct modbus_iface_param`
  - Device or device-like stubs.
- Structure members and member layouts matching the implementation.
- Logging macros:
  - `LOG_ERR`
  - `LOG_WRN`
  - `LOG_INF`
  - `LOG_DBG`
- Synchronization primitives and compatible operations:
  - `struct k_sem`
  - `struct k_mutex`
  - `k_sem_take`
  - `k_sem_give`
  - `k_mutex_lock`
  - `k_mutex_unlock`
- Any other macros, typedefs, constants, callbacks, or helper functions referenced by the source.

Check whether missing declarations would result in compiler errors, undeclared identifiers, incompatible-pointer warnings, or incorrect behavior.

### 3.3 Audit Assessment Report

Produce a clear assessment stating whether the current files are:

- `READY TO GO`, or
- Require specific corrections before refactoring.

For every required correction, provide:

1. The affected file.
2. The missing, incorrect, or incompatible symbol.
3. The expected definition or signature.
4. The reason the correction is necessary.
5. Any compatibility or behavioral risk.

Do not modify files during this pre-audit unless explicitly required by the implementation workflow. First document the findings.

## 4. Task 2: Refactor `modbus_serial.c` into `modbus_rtu.c`

Create or update:

```text
targets/modbus/modbus_rtu.c
```

Apply the following six modifications.

### 4.1 Header Isolation and Dependency Decoupling

Remove or comment out native Zephyr kernel, driver, and tree headers that are unavailable in the standalone build, including headers such as:

```c
/* #include <zephyr/kernel.h> */
/* #include <zephyr/device.h> */
/* #include <zephyr/drivers/uart.h> */
/* #include <zephyr/init.h> */
/* #include <zephyr/devicetree.h> */
/* #include <zephyr/logging/log.h> */
```

Replace them with standard C headers and the local compatibility header:

```c
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "zephyr_compat.h"
```

Use only the standard headers that are actually required by the resulting implementation.

Remove or replace all of the following where they depend on the Zephyr build system or device tree:

- `DT_*` macros.
- `DEVICE_DEFINE`.
- `SYS_INIT`.
- Runtime device-binding queries.
- Zephyr-only linker, initialization, and driver-registration mechanisms.

Preserve protocol behavior unless a requested modification explicitly changes it.

### 4.2 CRC Verification Bypass

Locate the Modbus RTU CRC calculation and validation logic, including functions such as `modbus_rtu_check_crc` or equivalent conditional blocks.

Modify the validation path so that malformed or mutated frames are not discarded solely because their CRC is invalid:

```c
static inline bool modbus_rtu_check_crc(const uint8_t *frame, uint16_t len)
{
    (void)frame;
    (void)len;
    return true; /* CRC bypass requested for fuzzing. */
}
```

If the original function has a different signature, preserve the required interface and apply the bypass without introducing incompatible declarations. Document clearly that this is a fuzzing-only behavior and must not be used in production firmware.

### 4.3 Replace Timers with Deterministic Polling or Synchronous Transitions

Locate all uses of `k_timer` and related operations, including:

- `k_timer_init`.
- `k_timer_start`.
- `k_timer_stop`.
- `k_timer_status_get`.
- Any callbacks implementing inter-character (`t1.5`) or inter-frame (`t3.5`) silence detection.

Remove kernel timer dependencies. Replace asynchronous timeout behavior with deterministic synchronous behavior suitable for a fuzzing harness:

1. Reset the parser before each injection.
2. Ingest the bounded input sequence byte-by-byte.
3. After ingestion completes, synchronously perform the equivalent frame-end transition.
4. Move the parser to the frame-received state or invoke the frame handler as appropriate.
5. Ensure the transition does not depend on wall-clock time, scheduler behavior, interrupts, or a running Zephyr kernel.

Do not silently omit frame completion if the original parser requires an explicit state transition.

### 4.4 Enforce a 256-Byte Frame Limit

Define the maximum frame size:

```c
#define MODBUS_MAX_FRAME_SIZE 256U
```

Ensure that every internal frame buffer and length counter respects this bound.

The fuzzing entry point must guard input ingestion. For input larger than 256 bytes, either:

- Clamp the input to the first 256 bytes, or
- Reject the excess input safely before it can affect the parser.

Select the behavior that best preserves deterministic exploration and document the choice. Never allow a write beyond the bounded frame buffer.

Also verify:

- Every increment of the frame length is bounds-checked.
- Length arithmetic cannot wrap before validation.
- Zero-length input is handled safely.
- A frame of exactly 256 bytes is handled without an off-by-one write.

### 4.5 Redirect Synchronization and Logging

Remove direct dependencies on RTOS synchronization primitives. If synchronization is not meaningful in the single-threaded fuzzing target, provide safe no-op compatibility operations for:

- `k_sem_take`.
- `k_sem_give`.
- `k_mutex_lock`.
- `k_mutex_unlock`.

Ensure that the no-op definitions preserve return types and accept the arguments used by the implementation without generating avoidable warnings.

Redirect diagnostic logging to standard I/O for triage:

```c
#define LOG_ERR(...) do { printf("[LOG_ERR] "); printf(__VA_ARGS__); } while (0)
#define LOG_WRN(...) do { printf("[LOG_WRN] "); printf(__VA_ARGS__); } while (0)
#define LOG_INF(...) do { printf("[LOG_INF] "); printf(__VA_ARGS__); } while (0)
#define LOG_DBG(...) do { } while (0)
```

Place these definitions in the compatibility layer when they are shared by multiple files; otherwise keep them local to the standalone target. Avoid macro definitions that produce malformed format strings or fail when called with no variadic arguments.

### 4.6 Export the Harness Entry Point

Expose the following C-callable function, adjusting only if the audited harness contract requires a compatible equivalent:

```c
void modbus_rtu_fuzz_inject(const uint8_t *data, size_t len);
```

The function must:

1. Validate the input pointer and length combination.
2. Reset the internal parser state and frame-buffer pointers.
3. Clamp or safely reject input beyond `MODBUS_MAX_FRAME_SIZE`.
4. Feed the accepted bytes sequentially through the Modbus RTU ingestion routines.
5. Synchronously signal frame completion.
6. Trigger the protocol parser or frame handler.
7. Leave the target ready for the next independent fuzzing iteration.

The function must not retain pointers into caller-owned input after it returns.

## 5. Task 3: Reconcile Supporting Files

After implementing the standalone target, re-audit:

- `targets/modbus/include/zephyr_compat.h`
- `targets/modbus/harness_modbus.c`
- `targets/modbus/modbus_rtu.c`

Provide an alignment summary that details every modification made to the compatibility header or harness, including:

- Changed function names or signatures.
- Added or removed declarations.
- Corrected structure definitions.
- Added reset behavior.
- Updated frame-size handling.
- Updated logging or synchronization stubs.
- Any changed ownership or lifetime assumptions.

## 6. Required Deliverables

Return the following in the final report:

1. A static audit assessment for `zephyr_compat.h` and `harness_modbus.c`.
2. A list of all API, type, structure, logging, synchronization, and state-reset mismatches.
3. The refactored `targets/modbus/modbus_rtu.c`.
4. Any corrected version of `targets/modbus/include/zephyr_compat.h`.
5. Any corrected version of `targets/modbus/harness_modbus.c`.
6. A description of the six requested modifications.
7. Native and cross-compilation results, including exact commands and diagnostics.
8. A final alignment summary explaining why the resulting files are compatible.

Do not claim that the target is ready or that a build succeeded unless the source files and commands have been directly inspected or executed.

## 7. Task 4: Create Actions.md file

create a markdown file, providing what changes has been made and what was the change and reason for that change, under relevant file. To grasp idea on the actions that have been made.
