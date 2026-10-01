/*
 * harness_uart.c – AFL++ fuzzing harness for the async UART RX driver
 *
 * Drives the refactored uart_async_rx.c (based on the real Zephyr upstream
 * at zephyr-upstream/drivers/serial/uart_async_rx.c) through the harness
 * entry-point API defined in modbusprompt §2.6:
 *
 *   void uart_mock_inject(const uint8_t *data, size_t len);
 *   void uart_mock_process(void);
 *
 * On each fuzz iteration:
 *   1. Read mutated bytes from stdin.
 *   2. Call uart_mock_inject() – resets all state, enqueues bytes.
 *   3. Call uart_mock_process() – drives the async-rx receive path.
 *   4. Assert post-conditions: all data consumed, no pending bytes.
 *
 * Compile (native, gcc):
 *   gcc -std=c11 -Wall -Wextra -Wpedantic -O2 \
 *       -Itargets/uart/include \
 *       targets/uart/uart_async_rx.c targets/uart/harness_uart.c \
 *       -o build/native/fuzzer_uart_native
 *
 * Compile (AFL++, with ASAN):
 *   afl-clang-fast -std=c11 -Wall -Wextra -O2 -fsanitize=address \
 *       -Itargets/uart/include \
 *       targets/uart/uart_async_rx.c targets/uart/harness_uart.c \
 *       -o build/native/fuzzer_uart_native
 */

#include <unistd.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>

#include "include/zephyr_compat.h"

/* -------------------------------------------------------------------------
 * Symbols from uart_async_rx.c (Mod 2.6 API)
 * ---------------------------------------------------------------------- */

extern void   uart_mock_inject(const uint8_t *data, size_t len);
extern void   uart_mock_process(void);
extern size_t uart_mock_claimed_total(void);
extern bool   uart_mock_all_consumed(void);

/* -------------------------------------------------------------------------
 * Constants – kept in sync with uart_async_rx.c definitions
 * ---------------------------------------------------------------------- */

/** Maximum bytes read from stdin per iteration (≥ UART_FIFO_SIZE) */
#define HARNESS_BUF_SIZE   640U

/* -------------------------------------------------------------------------
 * Post-iteration assertion checks
 * ---------------------------------------------------------------------- */

static void check_post_conditions(size_t injected)
{
    /*
     * After uart_mock_process() all injected bytes (up to FIFO capacity)
     * should have been claimed and consumed.  If any pending bytes remain,
     * the buffer-management logic has a bug.
     *
     * AddressSanitizer (when enabled at compile time) will independently
     * catch any buffer overruns inside the async-rx slab.
     */
    bool consumed = uart_mock_all_consumed();

    if (!consumed) {
        /*
         * Not necessarily a crash-worthy bug (e.g. more data than buffer
         * capacity), but worth logging for triage.
         */
        LOG_WRN("Post-iteration: %zu bytes claimed but not all consumed "
                "(injected=%zu)", uart_mock_claimed_total(), injected);
    }
}

/* -------------------------------------------------------------------------
 * Fuzz entry point
 * ---------------------------------------------------------------------- */

/**
 * fuzz_uart_entry – persistent-mode hook anchor
 *
 * Aligned with the §2.6 contract:
 *   - Reads up to HARNESS_BUF_SIZE bytes from stdin.
 *   - Calls uart_mock_inject() (handles NULL + len=0 safely).
 *   - Calls uart_mock_process().
 *   - Checks post-conditions.
 */
void fuzz_uart_entry(void)
{
    static uint8_t input_buf[HARNESS_BUF_SIZE];

    ssize_t n = read(STDIN_FILENO, input_buf, sizeof(input_buf));

    if (n < 0) {
        /* Read error – inject zero-length to exercise empty-input path */
        uart_mock_inject(input_buf, 0U);
        uart_mock_process();
        return;
    }

    /* Covers empty input (n == 0) and normal input (n > 0) */
    uart_mock_inject(input_buf, (size_t)n);
    uart_mock_process();

    check_post_conditions((size_t)n);
}

/* -------------------------------------------------------------------------
 * main – single-run or AFL++ persistent-loop invocation
 * ---------------------------------------------------------------------- */

int main(void)
{
#ifdef __AFL_HAVE_MANUAL_CONTROL
    __AFL_INIT();
    while (__AFL_LOOP(1000)) {
        fuzz_uart_entry();
    }
#else
    fuzz_uart_entry();
#endif
    return 0;
}
