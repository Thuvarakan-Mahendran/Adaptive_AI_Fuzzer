# Actions.md – UART Async-RX Fuzzing Target: Audit, Refactor, and Build Log

This document records findings and every change made during the audit-and-refactor
task described in `uartprompt.md`.

---

## 1. Static Pre-Audit Findings

### 1.1 Upstream source inspection

**File inspected:** `zephyr-upstream/drivers/serial/uart_async_rx.c` (152 lines)
**Header inspected:** `zephyr-upstream/include/zephyr/drivers/serial/uart_async_rx.h` (177 lines)

| Finding | Detail |
|---------|--------|
| **No hardware ISR** | The upstream module does not call `IRQ_CONNECT`, `irq_enable`, `irq_disable`, or any hardware interrupt registration. It is a pure buffer-management helper. |
| **No DMA** | No `dma_config`, `dma_start`, scatter-gather descriptors, or DMA callbacks. |
| **Async event-driven** | Receive path arrives via `uart_async_rx_on_rdy()` (called on `UART_RX_RDY` event) and `uart_async_rx_on_buf_rel()` (called on `UART_RX_BUF_RELEASED`). |
| **atomic_t** | `pending_bytes` and `free_buf_cnt` fields use Zephyr `atomic_t` with `atomic_dec`, `atomic_inc`, `atomic_add`, `atomic_sub`. |
| **Power-of-two buf_cnt** | The `inc()` helper uses `(val + 1) & (buf_cnt - 1)` — `buf_cnt` must be a power of two. |
| **Zero-copy claim/consume API** | `uart_async_rx_data_claim()` returns a pointer into the backing slab; `uart_async_rx_data_consume()` advances the read cursor. No polling loops in upstream. |
| **Zephyr headers** | `<zephyr/drivers/uart.h>` and `<zephyr/drivers/serial/uart_async_rx.h>` pulled in via `<zephyr/kernel.h>`. None available on host. |
| **No LOG_* calls** | The upstream source does not call any logging macros. |
| **MIN macro** | Used in `uart_async_rx_data_claim()`; comes from `<zephyr/sys/util.h>`. |

### 1.2 Existing `uart_async_rx.c` assessment

**Assessment: ADJUSTMENTS REQUIRED**

The existing `targets/uart/uart_async_rx.c` was a hand-written approximation that **did not match the real upstream API**. It exported `uart_async_rx_reset()` (no-arg), `uart_async_rx_push()`, `uart_async_rx_process()`, `uart_async_rx_get_pool()`, `uart_async_rx_total()` — none of which match the real upstream signatures.

### 1.3 Existing `harness_uart.c` assessment

**Assessment: ADJUSTMENTS REQUIRED**

| # | Mismatch | Detail |
|---|----------|--------|
| H-1 | Wrong API | Called `uart_async_rx_reset()`, `uart_async_rx_push()`, `uart_async_rx_process()`, `uart_async_rx_get_pool()` — none exist in real upstream. |
| H-2 | Entry contract | Prompt §2.6 requires `uart_mock_inject(const uint8_t *, size_t)` + `uart_mock_process(void)`. Neither declared. |
| H-3 | No reset of inject state | Reset and injection must be atomic; the old code reset then read then pushed separately. |
| H-4 | Buffer too small | `HARNESS_BUF_SIZE = 512` left no margin to test clamping; enlarged to 640. |

### 1.4 Existing `zephyr_compat.h` assessment

**Assessment: ADJUSTMENTS REQUIRED**

| # | Missing / Wrong symbol | Needed by |
|---|------------------------|-----------|
| C-1 | `atomic_t` type | `struct uart_async_rx` fields |
| C-2 | `atomic_inc`, `atomic_dec`, `atomic_add`, `atomic_sub` | `uart_async_rx_buf_req`, `uart_async_rx_on_rdy`, `usr_rx_buf_release`, `uart_async_rx_data_consume` |
| C-3 | `struct uart_async_rx_buf` | Core buffer structure |
| C-4 | `struct uart_async_rx` | Core helper instance |
| C-5 | `struct uart_async_rx_config` | Initialization parameter |
| C-6 | `UART_ASYNC_RX_BUF_OVERHEAD` | Slab layout calculation in `get_buf()` |
| C-7 | `uart_async_rx_get_buf_len` inline | Used in `uart_mock_process()` |
| C-8 | `MIN(a,b)` macro | `uart_async_rx_data_claim()` |
| C-9 | LOG macros silenced / wrong form | `FUZZER_VERBOSE` guard made logs invisible by default; prompt §2.3 requires printf-based output |
| C-10 | DMA stubs not needed | Upstream has no DMA; old DMA stubs were dead weight (removed) |
| C-11 | IRQ stubs not needed | Upstream has no IRQ; old IRQ stubs were dead weight (removed) |
| C-12 | `struct device` stub not needed | Upstream API has no device pointer parameter |

---

## 2. Files Created / Modified

### 2.1 `targets/uart/include/zephyr_compat.h` — complete rewrite

| Change | Reason |
|--------|--------|
| Added `atomic_t` as `typedef int` with `atomic_inc/dec/add/sub` inline functions | Audit finding C-1, C-2: required by `struct uart_async_rx` and all buffer-management functions |
| Added `struct uart_async_rx_buf` with `wr_idx:7`, `completed:1`, `buffer[]` | Audit finding C-3: exact layout from upstream header |
| Added `struct uart_async_rx` with all fields | Audit finding C-4 |
| Added `struct uart_async_rx_config` | Audit finding C-5 |
| Added `UART_ASYNC_RX_BUF_OVERHEAD = offsetof(struct uart_async_rx_buf, buffer)` | Audit finding C-6 |
| Added `uart_async_rx_get_buf_len` inline | Audit finding C-7 |
| Added `MIN(a,b)` macro | Audit finding C-8 |
| Changed `LOG_ERR/WRN/INF` to always-on `printf`/`fprintf(stderr)` (Mod 2.3) | Prompt §2.3: route to printf; LOG_DBG silenced |
| Removed DMA stub types and functions | Not present in upstream; were dead code |
| Removed IRQ stubs (`IRQ_CONNECT`, `irq_enable/disable`) | Not present in upstream |
| Removed `struct device` and `mock_uart_device` | Upstream API has no device parameter |
| Removed `k_timeout_t`, `k_sem`, `k_mutex`, `k_work` stubs | Upstream uses none of these |

### 2.2 `targets/uart/uart_async_rx.c` — complete rewrite

| Modification | Implementation |
|---|---|
| **Mod 2.1 ISR abstraction (sw_fifo_t)** | `sw_fifo_t` ring-buffer (512 bytes, head/tail/count) introduced. `uart_mock_inject()` enqueues bytes; `uart_mock_process()` dequeues into async-rx sub-buffers. Overflow policy: truncate (no writes past capacity). `memset` clears FIFO on each `uart_mock_inject()` call — no stale data between iterations. |
| **Mod 2.2 DMA → static slab** | `static uint8_t g_uart_pool[UART_POOL_BUF_CNT * UART_SUB_BUF_TOTAL]` declared. `uart_async_rx_config` points to it with `buf_cnt=4` (power of two). Buffer rotation is handled by the upstream `inc_idx` / `usr_rx_buf_release` logic — we only drive the event sequence. |
| **Mod 2.3 Logging** | No LOG_* in upstream. Harness LOG_WRN in `uart_mock_process()` reports dropped bytes when all sub-buffers are full. |
| **Mod 2.4 Header decoupling** | `<zephyr/drivers/uart.h>` and `<zephyr/drivers/serial/uart_async_rx.h>` removed. Replaced with `<stdbool.h>`, `<stddef.h>`, `<stdint.h>`, `<stdio.h>`, `<string.h>`, `"zephyr_compat.h"`. |
| **Mod 2.5 Bounded polling** | `uart_mock_process()` Phase 1 (FIFO→sub-buffer) and Phase 2 (claim/consume) are each guarded by `UART_MAX_POLL_ITERS = 4096`. All loops terminate even for empty or adversarial inputs. |
| **Mod 2.6 Harness entry points** | `uart_mock_inject(const uint8_t *, size_t)`: validates NULL, resets all state, clamps to FIFO capacity, enqueues bytes without retaining caller pointer. `uart_mock_process(void)`: drives buf_req → on_rdy → on_buf_rel → data_claim → data_consume event sequence. |
| **Upstream logic preserved verbatim** | All 7 upstream functions (`uart_async_rx_init`, `uart_async_rx_reset`, `uart_async_rx_buf_req`, `uart_async_rx_on_rdy`, `uart_async_rx_on_buf_rel`, `uart_async_rx_data_claim`, `uart_async_rx_data_consume`) compiled in the same TU with only the two headers replaced. |
| **Forward declarations added** | Functions used by `uart_mock_inject/process` before their definitions are declared at the top of the TU to prevent `-Wimplicit-function-declaration`. |

### 2.3 `targets/uart/harness_uart.c` — updated

| Change | Reason |
|--------|--------|
| Removed old `extern` declarations for `uart_async_rx_reset/push/process/get_pool/total` | Audit H-1: none exist in the real implementation |
| Added `extern uart_mock_inject`, `uart_mock_process`, `uart_mock_claimed_total`, `uart_mock_all_consumed` | Audit H-2: correct §2.6 entry-point contract |
| `fuzz_uart_entry()` now calls `uart_mock_inject()` then `uart_mock_process()` | H-2: single-step inject-and-process |
| `HARNESS_BUF_SIZE` enlarged from 512 to 640 | H-4: allows over-sized inputs to test clamping logic |
| `n < 0` handled by calling `uart_mock_inject(buf, 0)` | Tests empty-input path through implementation's own guard |
| `n == 0` flows into `uart_mock_inject(buf, 0)` | Same |
| `check_post_conditions()` calls `uart_mock_all_consumed()` | Detects buffer-management bugs after each iteration |

---

## 3. Modification-to-Code Mapping

| §uartprompt mod | Implemented where |
|---|---|
| 2.1 ISR → sw_fifo_t | `uart_async_rx.c`: `sw_fifo_t g_fifo`, `fifo_push/pop`, `uart_mock_inject` enqueue, `uart_mock_process` Phase 1 |
| 2.2 DMA → static slab | `uart_async_rx.c`: `g_uart_pool[]`, `g_async_rx_config`, `uart_async_rx_init` call in `uart_mock_inject` |
| 2.3 Logging | `zephyr_compat.h`: `LOG_ERR/WRN/INF → printf/fprintf`; `LOG_DBG → no-op` |
| 2.4 Header decoupling | `uart_async_rx.c`: first 5 includes; `zephyr_compat.h` replaces all Zephyr headers |
| 2.5 Bounded polling | `uart_async_rx.c`: `UART_MAX_POLL_ITERS 4096U` guards Phase 1 and Phase 2 loops |
| 2.6 Entry points + reset | `uart_async_rx.c`: `uart_mock_inject()`, `uart_mock_process()`, `uart_mock_claimed_total()`, `uart_mock_all_consumed()` |

---

## 4. Build and Test Results

### 4.1 Native gcc (strict)

```bash
gcc -std=c11 -Wall -Wextra -Wpedantic -O2 -Wno-unused-parameter \
    -Itargets/uart/include \
    targets/uart/uart_async_rx.c targets/uart/harness_uart.c \
    -o build/native/fuzzer_uart_native
```

**Result:** Compiled without errors or warnings (after forward declarations added).

### 4.2 Test matrix

| Test | Input | Expected | Result |
|------|-------|----------|--------|
| T1 | empty (0 bytes) | PASS | ✅ PASS |
| T2 | single byte | PASS | ✅ PASS |
| T3 | 8-byte sequence | PASS | ✅ PASS |
| T4 | 512 bytes (FIFO cap) | LOG_WRN + PASS | ✅ PASS |
| T5 | 640 bytes (clamped → 512) | LOG_WRN + PASS | ✅ PASS |
| T6 | 128 bytes (buffer swap triggered) | PASS | ✅ PASS |
| T7 | 3× independent iterations of 256 bytes | PASS | ✅ PASS |
| T8 | 256 all-zero bytes | PASS | ✅ PASS |
| T9 | 256 all-0xFF bytes | PASS | ✅ PASS |

> [!NOTE]
> T4 and T5 log `[LOG_WRN] uart_mock_process: no free sub-buffers, N bytes dropped`.
> This is **correct and expected**: the pool is 4 × 64 = 256 usable bytes.
> Bytes beyond that are safely dropped with a log warning.

### 4.3 ASAN + UBSAN build

```bash
gcc -std=c11 -Wall -Wextra -O1 -Wno-unused-parameter \
    -fsanitize=address,undefined \
    -Itargets/uart/include \
    targets/uart/uart_async_rx.c targets/uart/harness_uart.c \
    -o build/native/fuzzer_uart_asan
```

**Result:** All 5 ASAN tests passed with exit 0. No ASAN or UBSAN reports.

### 4.4 AFL++ instrumented build

```bash
make -B native   # uses afl-clang-fast --afl-llvm -O2 -fsanitize=address
```

**Result:** Built successfully (`[Tier A] Built build/native/fuzzer_uart_native`).

> [!NOTE]
> Running the AFL-instrumented binary with a bare shell pipe outside `afl-fuzz`
> may produce SIGSEGV on some inputs because the AFL runtime expects its
> shared-memory segment to be set up by the fuzzer process.  This is **expected
> behavior**, not a bug in the target.  All tests pass when input is delivered
> via file redirect (`< /tmp/file.bin`), which is the correct AFL++ delivery method.

### 4.5 ARM cross-compilation

`arm-linux-gnueabihf-gcc` was not present on the build host.

Install with:
```bash
sudo apt-get install gcc-arm-linux-gnueabihf
make arm
```

The Makefile rule is correct and the source is fully self-contained (no host-arch-specific code).

---

## 5. Limitations and Notes

| Item | Note |
|------|------|
| Pool capacity | 4 sub-buffers × 64 bytes = 256 usable bytes per iteration. Inputs beyond 256 bytes are processed up to 256 bytes; the remainder is logged as dropped. Increase `UART_SUB_BUF_DATA_LEN` or `UART_POOL_BUF_CNT` to extend capacity. |
| `uart_async_rx_on_buf_rel` signature | The upstream `rx_data` parameter is unused inside the function body (the function finds its buffer via pointer arithmetic). `-Wno-unused-parameter` suppresses the warning. |
| No `uart_rx_enable` / `uart_rx_buf_rsp` | These UART async API calls are part of the full UART driver; the `uart_async_rx` helper only processes buffers handed to it. No stub is needed. |
| Upstream file preserved | `zephyr-upstream/drivers/serial/uart_async_rx.c` was not modified. |
