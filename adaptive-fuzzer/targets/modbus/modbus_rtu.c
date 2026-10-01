/*
 * modbus_rtu.c – Zephyr Modbus RTU serial-frame handler, refactored for fuzzing
 *
 * Original source: Zephyr RTOS  subsys/modbus/modbus_serial.c
 * Copyright (c) 2020 PHYTEC Messtechnik GmbH
 * Copyright (c) 2021 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 *
 * ============================================================
 * Six modifications applied (per modbusprompt.md §4):
 *
 * Mod 4.1 – Header isolation / dependency decoupling
 *   All Zephyr kernel, driver, and devicetree headers removed.
 *   Replaced with <stdbool.h>, <stddef.h>, <stdint.h>, <stdio.h>,
 *   <string.h>, and the local "zephyr_compat.h".
 *   DT_*, DEVICE_DEFINE, SYS_INIT, and all driver-registration
 *   mechanisms removed.
 *
 * Mod 4.2 – CRC verification bypass
 *   modbus_rtu_check_crc() always returns true so mutated frames
 *   are not dropped before the protocol parser is exercised.
 *   *** FUZZING ONLY – must not be used in production firmware. ***
 *
 * Mod 4.3 – Timers replaced with synchronous transitions
 *   k_timer_init / k_timer_start / k_timer_stop and their callbacks
 *   (rtu_tmr_handler) are removed.  Frame completion is signalled
 *   synchronously by modbus_rtu_fuzz_inject() after all bytes are
 *   ingested, eliminating any wall-clock or scheduler dependency.
 *
 * Mod 4.4 – 256-byte frame cap
 *   MODBUS_MAX_FRAME_SIZE = 256 (defined in zephyr_compat.h).
 *   Every buffer declaration and length counter uses this limit.
 *   Input larger than 256 bytes is clamped to the first 256 bytes.
 *   A frame of exactly 256 bytes is handled without off-by-one.
 *
 * Mod 4.5 – Synchronization and logging redirected
 *   k_sem / k_mutex operations are no-ops (defined in zephyr_compat.h).
 *   LOG_* macros map to printf-based output (§4.5 spec).
 *
 * Mod 4.6 – Harness entry point exported
 *   void modbus_rtu_fuzz_inject(const uint8_t *data, size_t len)
 *   is the single public symbol consumed by harness_modbus.c.
 * ============================================================
 */

/* Mod 4.1 – Only standard C headers + local compat shim */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "zephyr_compat.h"

/* ===========================================================================
 * Module-level fuzzing context
 *
 * A single static modbus_context is used for all fuzz iterations.
 * It is fully reset by modbus_rtu_fuzz_inject() before each invocation
 * so no state leaks between iterations.
 * ========================================================================= */

static struct modbus_serial_config g_serial_cfg;
static struct modbus_context       g_ctx;

/* ===========================================================================
 * Mod 4.2 – CRC bypass
 *
 * Original Zephyr implementation compared calc_crc against the CRC stored
 * in the last two bytes of the frame.  For fuzzing we unconditionally
 * return true so mutated (invalid-CRC) frames exercise the full parser.
 * ========================================================================= */

/*
 * modbus_rtu_check_crc – CRC integrity gate (bypassed for fuzzing).
 *
 * FUZZING ONLY – always returns true.
 * DO NOT USE IN PRODUCTION.
 */
static inline bool modbus_rtu_check_crc(const uint8_t *frame, uint16_t len)
{
    (void)frame;
    (void)len;
    return true; /* CRC bypass requested for fuzzing. */
}

/* ===========================================================================
 * RTU receive-ADU parser
 *
 * Adapted from modbus_rtu_rx_adu() in modbus_serial.c.
 *
 * Changes versus upstream:
 *   - CONFIG_MODBUS_BUFFER_SIZE replaced by MODBUS_MAX_FRAME_SIZE.
 *   - sys_get_le16 available via zephyr_compat.h.
 *   - CRC comparison replaced by modbus_rtu_check_crc() (bypassed).
 *   - Removed dependency on ctx->cfg->uart_buf_ctr / uart_buf (we work
 *     directly from the harness-provided buffer via g_serial_cfg).
 * ========================================================================= */

static int modbus_rtu_rx_adu(struct modbus_context *ctx)
{
    struct modbus_serial_config *cfg = ctx->cfg;
    uint16_t crc_idx;
    uint8_t *data_ptr;

    /* Mod 4.4 – length bounds check */
    if (cfg->uart_buf_ctr < MODBUS_RTU_MIN_MSG_SIZE ||
        cfg->uart_buf_ctr > MODBUS_MAX_FRAME_SIZE) {
        LOG_WRN("Frame length error (%u bytes)", cfg->uart_buf_ctr);
        return -EMSGSIZE;
    }

    ctx->rx_adu.unit_id = cfg->uart_buf[0];
    ctx->rx_adu.fc      = cfg->uart_buf[1];
    data_ptr            = &cfg->uart_buf[2];

    /* Payload length: strip address (1), function code (1), CRC (2) */
    ctx->rx_adu.length  = cfg->uart_buf_ctr - 4U;

    /* CRC field is the last two bytes, little-endian */
    crc_idx             = cfg->uart_buf_ctr - (uint16_t)sizeof(uint16_t);
    ctx->rx_adu.crc     = sys_get_le16(&cfg->uart_buf[crc_idx]);

    /* Mod 4.4 – guard data copy against oversized payload */
    if (ctx->rx_adu.length > sizeof(ctx->rx_adu.data)) {
        LOG_WRN("Payload length %u exceeds ADU data buffer", ctx->rx_adu.length);
        ctx->rx_adu.length = (uint16_t)sizeof(ctx->rx_adu.data);
    }
    memcpy(ctx->rx_adu.data, data_ptr, ctx->rx_adu.length);

    /* Mod 4.2 – CRC check bypassed */
    if (!modbus_rtu_check_crc(cfg->uart_buf, cfg->uart_buf_ctr)) {
        LOG_WRN("CRC mismatch – dropping frame");
        return -EIO;
    }

    return 0;
}

/* ===========================================================================
 * Protocol-level frame dispatch
 *
 * In the real Zephyr stack the server_work work-queue item calls
 * modbus_serial_rx_adu() which feeds into the full Modbus server/client
 * request-response engine.  For fuzzing we replicate that path:
 *   1. Parse the raw RTU ADU.
 *   2. Log the decoded function code for triage.
 *   3. Any future deeper dispatch (FC handler table) goes here.
 * ========================================================================= */

static void dispatch_rx_adu(struct modbus_context *ctx)
{
    int rc = modbus_rtu_rx_adu(ctx);

    if (rc != 0) {
        LOG_WRN("modbus_rtu_rx_adu returned %d", rc);
        return;
    }

    LOG_DBG("RTU frame: unit_id=0x%02x fc=0x%02x payload_len=%u",
            ctx->rx_adu.unit_id,
            ctx->rx_adu.fc,
            ctx->rx_adu.length);

    /*
     * Extend here with a function-code dispatch table to exercise
     * deeper server logic (register reads, coil writes, etc.).
     */
}

/* ===========================================================================
 * Mod 4.6 – Public harness entry point
 *
 * void modbus_rtu_fuzz_inject(const uint8_t *data, size_t len)
 *
 * Contract (per modbusprompt.md §4.6):
 *   1. Validates input pointer and length.
 *   2. Resets internal parser state and buffer pointers.
 *   3. Clamps input to MODBUS_MAX_FRAME_SIZE (first 256 bytes kept).
 *   4. Feeds accepted bytes byte-by-byte through the RTU ingestion path.
 *   5. Synchronously signals frame completion (Mod 4.3).
 *   6. Triggers the protocol parser / frame handler.
 *   7. Leaves the target ready for the next iteration.
 *   8. Does not retain any pointer into caller-owned input after return.
 * ========================================================================= */

void modbus_rtu_fuzz_inject(const uint8_t *data, size_t len)
{
    /* Step 1 – Validate */
    if (data == NULL || len == 0U) {
        return;
    }

    /* Step 2 – Reset all context state so no values leak between iterations */
    memset(&g_serial_cfg, 0, sizeof(g_serial_cfg));
    memset(&g_ctx,        0, sizeof(g_ctx));

    g_serial_cfg.uart_buf_ptr = &g_serial_cfg.uart_buf[0];
    g_serial_cfg.uart_buf_ctr = 0U;
    g_ctx.cfg                 = &g_serial_cfg;
    g_ctx.mode                = MODBUS_MODE_RTU;
    g_ctx.client              = false;

    /* Step 3 – Clamp: accept at most MODBUS_MAX_FRAME_SIZE bytes.
     * Choice: clamp to first 256 bytes.  Rationale: preserves maximal
     * code-path diversity while preventing any write past the buffer. */
    if (len > MODBUS_MAX_FRAME_SIZE) {
        len = MODBUS_MAX_FRAME_SIZE;
    }

    /* Step 4 – Ingest bytes sequentially (mirrors ISR byte-by-byte path).
     * Mod 4.4 – every increment is bounds-checked; the buffer is
     * exactly MODBUS_MAX_FRAME_SIZE bytes so ctr can reach [0, 256]
     * without an off-by-one write. */
    for (size_t i = 0U; i < len; i++) {
        /* Guard: should never fire given the clamp above, but keep for safety */
        if (g_serial_cfg.uart_buf_ctr >= MODBUS_MAX_FRAME_SIZE) {
            break;
        }
        *g_serial_cfg.uart_buf_ptr++ = data[i];
        g_serial_cfg.uart_buf_ctr++;
    }

    /* Step 5 – Synchronous frame-completion transition (Mod 4.3).
     * The upstream rtu_tmr_handler fired k_work_submit(&ctx->server_work)
     * after the t3.5 silence timer expired.  We trigger the same
     * transition here, immediately after the last byte is ingested. */

    /* Step 6 – Trigger the protocol parser */
    dispatch_rx_adu(&g_ctx);

    /* Step 7 – The static context is already owned by this TU; no
     * caller-owned pointer is retained after this function returns.
     * The next call to modbus_rtu_fuzz_inject() will memset everything. */
}
