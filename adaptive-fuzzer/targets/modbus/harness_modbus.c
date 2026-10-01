/*
 * harness_modbus.c – AFL++ fuzzing harness for the Modbus RTU parser
 *
 * Connects the fuzzer's mutated byte stream to the standalone Modbus RTU
 * frame-processing logic exposed by modbus_rtu.c.
 *
 * Entry-point contract (modbusprompt.md §4.6):
 *   The harness calls  modbus_rtu_fuzz_inject(data, len)  which:
 *     - validates pointer and length,
 *     - resets all internal parser state,
 *     - clamps input to MODBUS_MAX_FRAME_SIZE (256 bytes),
 *     - ingests bytes byte-by-byte,
 *     - synchronously signals frame completion, and
 *     - triggers the protocol dispatcher.
 *
 * Compile for native fuzzing (inside adaptive-fuzzer/):
 *   afl-clang-fast -fsanitize=address -O2 \
 *       -Itargets/modbus/include \
 *       targets/modbus/modbus_rtu.c targets/modbus/harness_modbus.c \
 *       -o build/native/fuzzer_modbus_native
 */

#include <unistd.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>

/* Local Zephyr compatibility shim */
#include "include/zephyr_compat.h"

/* -------------------------------------------------------------------------
 * Public symbol from modbus_rtu.c (Mod 4.6 entry point)
 * ---------------------------------------------------------------------- */

extern void modbus_rtu_fuzz_inject(const uint8_t *data, size_t len);

/* -------------------------------------------------------------------------
 * Constants
 * ---------------------------------------------------------------------- */

/**
 * HARNESS_BUF_SIZE – read buffer for stdin input.
 *
 * Intentionally larger than MODBUS_MAX_FRAME_SIZE (256) so that we can
 * observe how the target handles over-sized inputs: modbus_rtu_fuzz_inject()
 * will clamp them internally to 256 bytes.
 */
#define HARNESS_BUF_SIZE  512U

/* -------------------------------------------------------------------------
 * Fuzz entry point
 * ---------------------------------------------------------------------- */

/**
 * fuzz_target_entry – persistent-mode hook anchor.
 *
 * Reads mutated bytes from stdin, then delegates to modbus_rtu_fuzz_inject()
 * which owns reset, ingestion, and dispatch.  Post-execution invariants are
 * checked after the call returns.
 */
void fuzz_target_entry(void)
{
    /*
     * Static buffer – allocated once per process, reused each iteration.
     * modbus_rtu_fuzz_inject() copies what it needs into its own internal
     * buffer, so no caller-owned pointer is retained after the call.
     */
    static uint8_t input_buf[HARNESS_BUF_SIZE];

    /* Read one test case from the AFL++ mutated input channel (stdin) */
    ssize_t n = read(STDIN_FILENO, input_buf, sizeof(input_buf));
    if (n <= 0) {
        /*
         * Zero-length input: pass len=0 to let the implementation handle
         * it (early return inside modbus_rtu_fuzz_inject).
         */
        modbus_rtu_fuzz_inject(input_buf, 0U);
        return;
    }

    /* Invoke the harness entry point – reset + ingest + dispatch */
    modbus_rtu_fuzz_inject(input_buf, (size_t)n);

    /*
     * Post-execution invariant check.
     *
     * The internal context is opaque from here, but AddressSanitizer
     * (enabled at compile-time with -fsanitize=address) will catch any
     * heap/stack overflows.  Protocol-level assertions can be added via
     * an accessor once the context is exposed.
     *
     * Example (extend when modbus_rtu_get_context() is available):
     *
     *   const struct modbus_context *ctx = modbus_rtu_get_context();
     *   assert(ctx->cfg->uart_buf_ctr <= MODBUS_MAX_FRAME_SIZE);
     */
}

/* -------------------------------------------------------------------------
 * main – single-run or persistent-loop invocation
 * ---------------------------------------------------------------------- */

int main(void)
{
#ifdef __AFL_HAVE_MANUAL_CONTROL
    /*
     * AFL++ persistent mode: fork once, iterate up to 1 000 times, then
     * respawn.  Amortises fork() cost → dramatically higher throughput.
     */
    __AFL_INIT();
    while (__AFL_LOOP(1000)) {
        fuzz_target_entry();
    }
#else
    /* Standard single-run mode – useful for debugging with a fixed input. */
    fuzz_target_entry();
#endif
    return 0;
}
