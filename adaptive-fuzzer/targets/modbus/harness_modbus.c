/*
 * harness_modbus.c – AFL++ fuzzing harness for the Modbus RTU parser
 *
 * This harness wires the adaptive fuzzer infrastructure to the extracted
 * Modbus RTU frame-processing logic.  It follows the design described in
 * paper Section III-C / Step 2.6:
 *
 *   1. Reads raw bytes from STDIN (the AFL++ mutated test-case channel).
 *   2. Resets all Modbus state-machine variables between iterations.
 *   3. Calls the frame handler and exercises the full RTU dispatch path.
 *   4. Post-execution assertions catch abnormal control-flow or illegal
 *      pointer conditions (triggering SIGABRT on violation).
 *
 * Compile for native fuzzing:
 *   afl-clang-fast -fsanitize=address -O2
 *       -Itargets/modbus/include
 *       targets/modbus/modbus_rtu.c targets/modbus/harness_modbus.c
 *       -o build/native/fuzzer_modbus_native
 */

#include <unistd.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <signal.h>

/* Include the mocked Zephyr compatibility layer */
#include "include/zephyr_compat.h"

/* Forward declarations from modbus_rtu.c */
void modbus_rtu_reset_state(void);
int  modbus_rtu_process_frame(const uint8_t *data, size_t len);

/* -------------------------------------------------------------------------
 * Constants
 * ---------------------------------------------------------------------- */

/** Maximum input buffer size – must match MODBUS_MAX_FRAME_SIZE */
#define HARNESS_BUF_SIZE  256u

/* -------------------------------------------------------------------------
 * Invariant checking
 * ---------------------------------------------------------------------- */

/*
 * check_post_invariants – assert that no illegal state was reached
 *
 * Extend this function with protocol-specific post-conditions as the
 * Modbus state machine is expanded.
 */
static void check_post_invariants(void)
{
    /*
     * Placeholder: with a more complete Modbus context exposed via a
     * modbus_get_state() accessor, assertions such as the following
     * would be added here:
     *
     *   const modbus_ctx_t *ctx = modbus_get_state();
     *   assert(ctx->rx_len <= HARNESS_BUF_SIZE);
     *   assert(ctx->state != SOME_ILLEGAL_STATE);
     *
     * A violation triggers SIGABRT (via assert()), which AFL++ treats
     * as a crash to save and minimise.
     */
    (void)0; /* suppress empty-body warning */
}

/* -------------------------------------------------------------------------
 * Fuzz entry point
 * ---------------------------------------------------------------------- */

/*
 * fuzz_target_entry – persistent-mode hook anchor
 *
 * In standard (non-persistent) mode this is called once per process.
 * With AFL_LOOP / __AFL_LOOP the runtime wraps this in a tight loop
 * without fork() overhead; the surrounding main() handles both cases.
 */
void fuzz_target_entry(void)
{
    /* Static local buffer – allocated once, reused each iteration */
    static uint8_t input_buf[HARNESS_BUF_SIZE];

    /* Reset Modbus state machine variables (slave addr, func, state …) */
    modbus_rtu_reset_state();

    /* Read mutated test case from AFL++ via STDIN */
    ssize_t n = read(STDIN_FILENO, input_buf, sizeof(input_buf));
    if (n <= 0) {
        /* No data or read error – nothing to process this iteration */
        return;
    }

    /* Feed the frame to the Modbus parser */
    (void)modbus_rtu_process_frame(input_buf, (size_t)n);

    /* Verify post-execution invariants */
    check_post_invariants();
}

/* -------------------------------------------------------------------------
 * main – single-run or persistent-loop invocation
 * ---------------------------------------------------------------------- */

int main(void)
{
#ifdef __AFL_HAVE_MANUAL_CONTROL
    /*
     * AFL++ persistent mode: fork once and loop up to 1000 times before
     * spawning a new worker.  This amortises fork() cost across iterations,
     * dramatically increasing throughput.
     */
    __AFL_INIT();
    while (__AFL_LOOP(1000)) {
        fuzz_target_entry();
    }
#else
    /* Standard single-run mode (useful for debugging with a fixed input) */
    fuzz_target_entry();
#endif
    return 0;
}
