/*
 * harness_uart.c – AFL++ fuzzing harness for the async UART RX driver
 *
 * Wires the adaptive fuzzer to the software-simulated async UART reception
 * path (uart_async_rx.c).  Follows paper Section III-D / Step 3.6:
 *
 *   1. Reads mutated input bytes from STDIN.
 *   2. Resets all driver state (static pools, FIFO r/w pointers, state machine).
 *   3. Pushes fuzz data directly into the simulated software FIFO.
 *   4. Triggers the receive processing routine.
 *   5. Asserts that the static reception pools did not overflow.
 *
 * Compile for native fuzzing:
 *   afl-clang-fast -fsanitize=address -O2
 *       -Itargets/uart/include
 *       targets/uart/uart_async_rx.c targets/uart/harness_uart.c
 *       -o build/native/fuzzer_uart_native
 */

#include <unistd.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>

/* Include the mocked Zephyr compatibility layer */
#include "include/zephyr_compat.h"

/* Forward declarations from uart_async_rx.c */
void           uart_async_rx_reset(void);
size_t         uart_async_rx_push(const uint8_t *data, size_t len);
size_t         uart_async_rx_process(void);
const uint8_t *uart_async_rx_get_pool(uint32_t idx, uint32_t *out_used);
uint32_t       uart_async_rx_total(void);

/* -------------------------------------------------------------------------
 * Constants
 * ---------------------------------------------------------------------- */

/** Input buffer size – should not exceed the FIFO capacity in uart_async_rx.c */
#define HARNESS_BUF_SIZE      512u

/** Reception pool buffer size – must match UART_POOL_BUF_SIZE */
#define POOL_BUF_SIZE         256u

/** Number of reception pools – must match UART_POOL_COUNT */
#define POOL_COUNT            2u

/* -------------------------------------------------------------------------
 * Pool overflow / boundary assertions
 * ---------------------------------------------------------------------- */

/*
 * check_pool_boundaries – verify that neither reception pool overflowed
 *
 * Asserts that the `used` count for every pool does not exceed the pool's
 * allocated size, catching off-by-one and buffer-boundary overwrites.
 * Triggers SIGABRT (via assert()) if a violation is detected.
 */
static void check_pool_boundaries(void)
{
    for (uint32_t i = 0; i < POOL_COUNT; i++) {
        uint32_t used = 0;
        const uint8_t *pool_data = uart_async_rx_get_pool(i, &used);
        assert(pool_data != NULL);
        assert(used <= POOL_BUF_SIZE);
    }
}

/* -------------------------------------------------------------------------
 * Fuzz entry point
 * ---------------------------------------------------------------------- */

/*
 * fuzz_uart_entry – persistent-mode hook anchor
 */
void fuzz_uart_entry(void)
{
    /* Static local buffer – allocated once, reused each iteration */
    static uint8_t input_buf[HARNESS_BUF_SIZE];

    /* Step 1: Reset all driver state between iterations
     *  – Zeros static reception pools
     *  – Zeroes FIFO read/write pointers
     *  – Resets receiver state machine
     */
    uart_async_rx_reset();

    /* Step 2: Read mutated test case from AFL++ via STDIN */
    ssize_t n = read(STDIN_FILENO, input_buf, sizeof(input_buf));
    if (n <= 0) {
        return; /* nothing to do */
    }

    /* Step 3: Inject fuzz bytes into the simulated FIFO
     *  – Mimics incoming peripheral stream bytes arriving via UART RX
     */
    size_t pushed = uart_async_rx_push(input_buf, (size_t)n);
    (void)pushed; /* partial push is not an error; FIFO may be full */

    /* Step 4: Trigger the driver's receive-processing routine
     *  – Transfers bytes from the FIFO to the static pool (replacing
     *    the ISR / DMA path)
     */
    (void)uart_async_rx_process();

    /* Step 5: Assert pool boundaries (detect overflow / off-by-one) */
    check_pool_boundaries();
}

/* -------------------------------------------------------------------------
 * main – single-run or persistent-loop invocation
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
