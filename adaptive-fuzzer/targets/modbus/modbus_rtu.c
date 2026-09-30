/*
 * modbus_rtu.c – Zephyr Modbus RTU parser adapted for fuzzing
 *
 * Source: Extracted and adapted from Zephyr RTOS
 *         subsys/modbus/modbus_rtu.c (Apache-2.0 / Zephyr Project)
 *
 * Key adaptations (per paper Section III-C):
 *   1. Zephyr RTOS dependencies replaced by include/zephyr_compat.h stubs.
 *   2. CRC validation bypassed – modbus_rtu_check_crc() always returns true.
 *   3. Hardware timer callbacks (t1_5 / t3_5 silent-interval timers) removed;
 *      state transitions are now synchronous (evaluated when the input buffer
 *      is exhausted or an explicit end-of-frame sentinel is seen).
 *   4. Frame size hard-capped at MODBUS_MAX_FRAME_SIZE (256 bytes).
 */

#include "include/zephyr_compat.h"
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

/* -------------------------------------------------------------------------
 * Frame-size guard (Step 2.5)
 * ---------------------------------------------------------------------- */

#define MODBUS_MAX_FRAME_SIZE  256u

/* -------------------------------------------------------------------------
 * Modbus RTU state machine states
 * ---------------------------------------------------------------------- */

typedef enum {
    MODBUS_STATE_IDLE = 0,
    MODBUS_STATE_RECEIVING,
    MODBUS_STATE_PROCESSING,
    MODBUS_STATE_DONE,
    MODBUS_STATE_ERROR,
} modbus_state_t;

/* -------------------------------------------------------------------------
 * Internal Modbus context (state machine variables)
 * ---------------------------------------------------------------------- */

typedef struct {
    modbus_state_t  state;
    uint8_t         slave_addr;
    uint8_t         func_code;
    uint16_t        transaction_id;  /* used in Modbus TCP mode; zeroed for RTU */
    uint8_t         rx_buf[MODBUS_MAX_FRAME_SIZE];
    size_t          rx_len;
    bool            frame_error;
} modbus_ctx_t;

/* Module-level context – reset between fuzz iterations by the harness */
static modbus_ctx_t g_ctx;

/* -------------------------------------------------------------------------
 * CRC-16 (Modbus) – kept for reference; bypassed during fuzzing
 * (Step 2.3: force the check to always return true)
 * ---------------------------------------------------------------------- */

static uint16_t crc16_modbus(const uint8_t *data, size_t len)
{
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= (uint16_t)data[i];
        for (int j = 0; j < 8; j++) {
            if (crc & 0x0001) {
                crc = (crc >> 1) ^ 0xA001;
            } else {
                crc >>= 1;
            }
        }
    }
    return crc;
}

/*
 * modbus_rtu_check_crc – CRC integrity gate
 *
 * FUZZER ADAPTATION: always returns true so the fuzzer can reach deep
 * command-handling states without having to produce a valid 16-bit CRC.
 * The real CRC computation is preserved above for reference.
 */
static bool modbus_rtu_check_crc(const uint8_t *frame, size_t len)
{
    (void)crc16_modbus; /* suppress unused-function warning */
    (void)frame;
    (void)len;
    /* Bypass: pretend every CRC is valid */
    return true;
}

/* -------------------------------------------------------------------------
 * PDU field accessors
 * ---------------------------------------------------------------------- */

static inline uint8_t  frame_slave_addr(const uint8_t *f) { return f[0]; }
static inline uint8_t  frame_func_code (const uint8_t *f) { return f[1]; }
static inline uint16_t frame_be16(const uint8_t *f, size_t off)
{
    return (uint16_t)((f[off] << 8) | f[off + 1]);
}

/* -------------------------------------------------------------------------
 * Function-code dispatch table
 *
 * Each handler receives the full frame and its length.  Handlers may update
 * g_ctx to reflect post-processing state.
 * ---------------------------------------------------------------------- */

static void handle_read_coils(const uint8_t *frame, size_t len)
{
    if (len < 6) { g_ctx.frame_error = true; return; }
    uint16_t start_addr = frame_be16(frame, 2);
    uint16_t quantity   = frame_be16(frame, 4);
    LOG_DBG("READ_COILS addr=0x%04x qty=%u", start_addr, quantity);
    (void)start_addr; (void)quantity;
}

static void handle_read_discrete_inputs(const uint8_t *frame, size_t len)
{
    if (len < 6) { g_ctx.frame_error = true; return; }
    uint16_t start_addr = frame_be16(frame, 2);
    uint16_t quantity   = frame_be16(frame, 4);
    LOG_DBG("READ_DISCRETE_INPUTS addr=0x%04x qty=%u", start_addr, quantity);
    (void)start_addr; (void)quantity;
}

static void handle_read_holding_registers(const uint8_t *frame, size_t len)
{
    if (len < 6) { g_ctx.frame_error = true; return; }
    uint16_t start_addr = frame_be16(frame, 2);
    uint16_t quantity   = frame_be16(frame, 4);
    LOG_DBG("READ_HOLDING_REGS addr=0x%04x qty=%u", start_addr, quantity);
    (void)start_addr; (void)quantity;
}

static void handle_read_input_registers(const uint8_t *frame, size_t len)
{
    if (len < 6) { g_ctx.frame_error = true; return; }
    uint16_t start_addr = frame_be16(frame, 2);
    uint16_t quantity   = frame_be16(frame, 4);
    LOG_DBG("READ_INPUT_REGS addr=0x%04x qty=%u", start_addr, quantity);
    (void)start_addr; (void)quantity;
}

static void handle_write_single_coil(const uint8_t *frame, size_t len)
{
    if (len < 6) { g_ctx.frame_error = true; return; }
    uint16_t coil_addr  = frame_be16(frame, 2);
    uint16_t coil_value = frame_be16(frame, 4);
    LOG_DBG("WRITE_SINGLE_COIL addr=0x%04x val=0x%04x", coil_addr, coil_value);
    (void)coil_addr; (void)coil_value;
}

static void handle_write_single_register(const uint8_t *frame, size_t len)
{
    if (len < 6) { g_ctx.frame_error = true; return; }
    uint16_t reg_addr  = frame_be16(frame, 2);
    uint16_t reg_value = frame_be16(frame, 4);
    LOG_DBG("WRITE_SINGLE_REG addr=0x%04x val=0x%04x", reg_addr, reg_value);
    (void)reg_addr; (void)reg_value;
}

static void handle_write_multiple_coils(const uint8_t *frame, size_t len)
{
    if (len < 7) { g_ctx.frame_error = true; return; }
    uint16_t start_addr  = frame_be16(frame, 2);
    uint16_t quantity    = frame_be16(frame, 4);
    uint8_t  byte_count  = frame[6];
    LOG_DBG("WRITE_MULTIPLE_COILS addr=0x%04x qty=%u byte_count=%u",
            start_addr, quantity, byte_count);
    if (len < (size_t)(7 + byte_count)) { g_ctx.frame_error = true; return; }
    (void)start_addr; (void)quantity; (void)byte_count;
}

static void handle_write_multiple_registers(const uint8_t *frame, size_t len)
{
    if (len < 7) { g_ctx.frame_error = true; return; }
    uint16_t start_addr  = frame_be16(frame, 2);
    uint16_t quantity    = frame_be16(frame, 4);
    uint8_t  byte_count  = frame[6];
    LOG_DBG("WRITE_MULTIPLE_REGS addr=0x%04x qty=%u byte_count=%u",
            start_addr, quantity, byte_count);
    if (len < (size_t)(7 + byte_count)) { g_ctx.frame_error = true; return; }
    (void)start_addr; (void)quantity; (void)byte_count;
}

static void handle_mask_write_register(const uint8_t *frame, size_t len)
{
    if (len < 8) { g_ctx.frame_error = true; return; }
    uint16_t ref_addr  = frame_be16(frame, 2);
    uint16_t and_mask  = frame_be16(frame, 4);
    uint16_t or_mask   = frame_be16(frame, 6);
    LOG_DBG("MASK_WRITE_REG addr=0x%04x and=0x%04x or=0x%04x",
            ref_addr, and_mask, or_mask);
    (void)ref_addr; (void)and_mask; (void)or_mask;
}

static void handle_read_write_multiple_registers(const uint8_t *frame, size_t len)
{
    if (len < 11) { g_ctx.frame_error = true; return; }
    uint16_t read_addr   = frame_be16(frame, 2);
    uint16_t read_count  = frame_be16(frame, 4);
    uint16_t write_addr  = frame_be16(frame, 6);
    uint16_t write_count = frame_be16(frame, 8);
    uint8_t  byte_count  = frame[10];
    LOG_DBG("READ_WRITE_MULT_REGS r_addr=0x%04x r_cnt=%u w_addr=0x%04x "
            "w_cnt=%u byte_count=%u",
            read_addr, read_count, write_addr, write_count, byte_count);
    if (len < (size_t)(11 + byte_count)) { g_ctx.frame_error = true; return; }
    (void)read_addr; (void)read_count; (void)write_addr;
    (void)write_count; (void)byte_count;
}

static void handle_unknown_function(const uint8_t *frame, size_t len)
{
    LOG_WRN("Unknown function code 0x%02x (len=%zu)", frame[1], len);
    g_ctx.frame_error = true;
}

/* -------------------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------------- */

/*
 * modbus_rtu_reset_state – reset all state machine variables
 *
 * Called by the harness between fuzz iterations to prevent state leakage.
 */
void modbus_rtu_reset_state(void)
{
    memset(&g_ctx, 0, sizeof(g_ctx));
    g_ctx.state = MODBUS_STATE_IDLE;
}

/*
 * modbus_rtu_process_frame – main entry point for the frame parser
 *
 * @param data   Pointer to raw RTU frame bytes
 * @param len    Number of bytes in the frame
 *
 * Returns 0 on success, -1 on hard error (frame too large, NULL pointer).
 *
 * FUZZER ADAPTATION (Step 2.4):
 *   - No hardware timer callbacks are invoked.
 *   - The function processes the entire supplied buffer synchronously and
 *     transitions to MODBUS_STATE_DONE upon exhausting it.
 */
int modbus_rtu_process_frame(const uint8_t *data, size_t len)
{
    /* Guard: null or zero-length */
    if (!data || len == 0) {
        return -1;
    }

    /* Guard: frame-size cap (Step 2.5) */
    if (len > MODBUS_MAX_FRAME_SIZE) {
        LOG_WRN("Frame too large (%zu > %u), truncating", len, MODBUS_MAX_FRAME_SIZE);
        len = MODBUS_MAX_FRAME_SIZE;
    }

    /* Minimum Modbus RTU frame: 1-byte address + 1-byte func + 2-byte CRC = 4 */
    if (len < 4) {
        g_ctx.state = MODBUS_STATE_ERROR;
        return -1;
    }

    /* Copy into internal buffer and update state */
    memcpy(g_ctx.rx_buf, data, len);
    g_ctx.rx_len = len;
    g_ctx.state  = MODBUS_STATE_RECEIVING;

    /* CRC check – bypassed for fuzzing (Step 2.3) */
    if (!modbus_rtu_check_crc(g_ctx.rx_buf, g_ctx.rx_len)) {
        LOG_WRN("CRC mismatch – dropping frame");
        g_ctx.state = MODBUS_STATE_ERROR;
        return -1;
    }

    /* Extract address and function code */
    g_ctx.slave_addr     = frame_slave_addr(g_ctx.rx_buf);
    g_ctx.func_code      = frame_func_code(g_ctx.rx_buf);
    g_ctx.transaction_id = 0; /* RTU has no transaction ID */
    g_ctx.frame_error    = false;

    g_ctx.state = MODBUS_STATE_PROCESSING;

    LOG_DBG("Processing frame: slave=0x%02x fc=0x%02x len=%zu",
            g_ctx.slave_addr, g_ctx.func_code, g_ctx.rx_len);

    /* Dispatch on function code */
    switch (g_ctx.func_code) {
    case 0x01: handle_read_coils(g_ctx.rx_buf, g_ctx.rx_len);                  break;
    case 0x02: handle_read_discrete_inputs(g_ctx.rx_buf, g_ctx.rx_len);        break;
    case 0x03: handle_read_holding_registers(g_ctx.rx_buf, g_ctx.rx_len);      break;
    case 0x04: handle_read_input_registers(g_ctx.rx_buf, g_ctx.rx_len);        break;
    case 0x05: handle_write_single_coil(g_ctx.rx_buf, g_ctx.rx_len);           break;
    case 0x06: handle_write_single_register(g_ctx.rx_buf, g_ctx.rx_len);       break;
    case 0x0F: handle_write_multiple_coils(g_ctx.rx_buf, g_ctx.rx_len);        break;
    case 0x10: handle_write_multiple_registers(g_ctx.rx_buf, g_ctx.rx_len);    break;
    case 0x16: handle_mask_write_register(g_ctx.rx_buf, g_ctx.rx_len);         break;
    case 0x17: handle_read_write_multiple_registers(g_ctx.rx_buf, g_ctx.rx_len); break;
    default:   handle_unknown_function(g_ctx.rx_buf, g_ctx.rx_len);            break;
    }

    /* Synchronous end-of-frame transition (replaces t3_5 timer, Step 2.4) */
    g_ctx.state = g_ctx.frame_error ? MODBUS_STATE_ERROR : MODBUS_STATE_DONE;
    return 0;
}
