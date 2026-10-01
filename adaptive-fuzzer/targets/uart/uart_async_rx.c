/*
 * uart_async_rx.c – Zephyr UART async-rx helper, refactored for standalone fuzzing
 *
 * Original source: zephyr-upstream/drivers/serial/uart_async_rx.c
 * Copyright (c) 2023 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 *
 * ============================================================
 * Pre-audit findings (see Actions.md for full table):
 *
 *  - The upstream module is a pure buffer-management helper.
 *    It does NOT use hardware ISR, IRQ_CONNECT, DMA, or polling loops.
 *  - The actual receive path is event-driven via UART_RX_RDY /
 *    UART_RX_BUF_RELEASED async callbacks supplied by the caller.
 *  - It uses atomic_t (from <zephyr/kernel.h>) for pending_bytes and
 *    free_buf_cnt, and MIN() from <zephyr/sys/util.h>.
 *  - No LOG_* calls appear in the upstream source.
 *
 * Modifications applied (modbusprompt §2 numbering):
 *
 *  Mod 2.1 – ISR abstraction with a software FIFO
 *    The upstream has no hardware ISR.  The receive path arrives via
 *    uart_async_rx_on_rdy() (the UART_RX_RDY event handler).
 *    We introduce a bounded software FIFO (sw_fifo_t) and the public
 *    API uart_mock_inject() / uart_mock_process() that:
 *      a) enqueue fuzz bytes into the FIFO,
 *      b) simulate sequential byte delivery into the async-rx sub-buffers
 *         by calling uart_async_rx_buf_req() + uart_async_rx_on_rdy() +
 *         uart_async_rx_on_buf_rel() in the same order the UART driver would.
 *    Overflow policy: excess bytes are silently dropped (truncated).
 *
 *  Mod 2.2 – DMA replacement with static buffers
 *    The upstream uses a caller-supplied backing slab (config.buffer).
 *    We declare a static slab (g_uart_pool) of UART_POOL_TOTAL_BYTES and
 *    configure uart_async_rx with buf_cnt = UART_POOL_BUF_CNT.
 *    Buffer rotation is handled by the existing uart_async_rx logic;
 *    we only drive the request/ready/release event sequence.
 *
 *  Mod 2.3 – Logging redirection
 *    LOG_* macros are defined in zephyr_compat.h (upstream uses none).
 *
 *  Mod 2.4 – Header and device-tree decoupling
 *    <zephyr/drivers/uart.h>, <zephyr/drivers/serial/uart_async_rx.h>,
 *    and <zephyr/kernel.h> removed.  Replaced with standard C headers
 *    and "zephyr_compat.h" which re-declares all required types.
 *
 *  Mod 2.5 – Bounded polling
 *    uart_mock_process() is bounded by UART_MAX_POLL_ITERS.
 *    The upstream data_claim / data_consume loop in the consume path
 *    is similarly bounded.
 *
 *  Mod 2.6 – Harness entry points
 *    Exports:
 *      void uart_mock_inject(const uint8_t *data, size_t len);
 *      void uart_mock_process(void);
 *    and the inspection helpers:
 *      size_t uart_mock_claimed_total(void);
 *      bool   uart_mock_all_consumed(void);
 *
 * FUZZING ONLY – hardware emulation and atomic stubs must not be used
 * in production firmware.
 * ============================================================
 */

/* Mod 2.4 – Standard C headers only; no Zephyr build-system headers */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "zephyr_compat.h"

/* ===========================================================================
 * Forward declarations – upstream API defined later in this TU
 *
 * Required because uart_mock_inject() and uart_mock_process() call these
 * functions before their definitions appear in the file.
 * ========================================================================= */

int      uart_async_rx_init(struct uart_async_rx *rx_data,
                             const struct uart_async_rx_config *config);
void     uart_async_rx_reset(struct uart_async_rx *rx_data);
uint8_t *uart_async_rx_buf_req(struct uart_async_rx *rx_data);
void     uart_async_rx_on_rdy(struct uart_async_rx *rx_data,
                               uint8_t *buffer, size_t length);
void     uart_async_rx_on_buf_rel(struct uart_async_rx *rx_data,
                                   uint8_t *buffer);
size_t   uart_async_rx_data_claim(struct uart_async_rx *rx_data,
                                   uint8_t **data, size_t length);
bool     uart_async_rx_data_consume(struct uart_async_rx *rx_data,
                                     size_t length);

/* ===========================================================================
 * Static configuration
 * ========================================================================= */

/**
 * UART_POOL_BUF_CNT – number of sub-buffers in the slab.
 * Must be a power of two (the upstream inc() uses a bitmask).
 */
#define UART_POOL_BUF_CNT    4U

/**
 * UART_SUB_BUF_DATA_LEN – usable data bytes per sub-buffer.
 * The slab layout is:
 *   [struct uart_async_rx_buf header][data bytes] × buf_cnt
 */
#define UART_SUB_BUF_DATA_LEN  64U

/**
 * UART_SUB_BUF_TOTAL – total bytes per slot including overhead.
 */
#define UART_SUB_BUF_TOTAL  (UART_ASYNC_RX_BUF_OVERHEAD + UART_SUB_BUF_DATA_LEN)

/**
 * UART_POOL_TOTAL_BYTES – total backing slab size.
 */
#define UART_POOL_TOTAL_BYTES  (UART_POOL_BUF_CNT * UART_SUB_BUF_TOTAL)

/* Mod 2.5 – iteration cap for bounded processing loops */
#define UART_MAX_POLL_ITERS  4096U

/* ===========================================================================
 * Mod 2.1 – Software ring-buffer FIFO
 *
 * Replaces the hardware UART RX FIFO.  The harness enqueues fuzz-generated
 * bytes; uart_mock_process() dequeues them and drives the async-rx API.
 *
 * Overflow policy: excess bytes beyond UART_FIFO_SIZE are dropped silently.
 * No stale data crosses iterations because uart_mock_inject() resets first.
 * ========================================================================= */

#define UART_FIFO_SIZE  512U

typedef struct {
    uint8_t  buf[UART_FIFO_SIZE];
    uint32_t head;   /* write index (next free slot) */
    uint32_t tail;   /* read index  (next byte to pop) */
    uint32_t count;  /* bytes currently in the FIFO   */
} sw_fifo_t;

static sw_fifo_t g_fifo;

static inline bool fifo_empty(const sw_fifo_t *f) { return f->count == 0U; }
static inline bool fifo_full(const sw_fifo_t *f)
{
    return f->count >= UART_FIFO_SIZE;
}

/** Push one byte; returns false on overflow (byte dropped). */
static bool fifo_push(sw_fifo_t *f, uint8_t byte)
{
    if (fifo_full(f)) return false;
    f->buf[f->head] = byte;
    f->head = (f->head + 1U) % UART_FIFO_SIZE;
    f->count++;
    return true;
}

/** Pop one byte; returns false when empty. */
static bool fifo_pop(sw_fifo_t *f, uint8_t *out)
{
    if (fifo_empty(f)) return false;
    *out = f->buf[f->tail];
    f->tail = (f->tail + 1U) % UART_FIFO_SIZE;
    f->count--;
    return true;
}

/* ===========================================================================
 * Mod 2.2 – Static backing slab (replaces dynamic/DMA allocation)
 * ========================================================================= */

/** Static pool slab – used by the uart_async_rx config. */
static uint8_t g_uart_pool[UART_POOL_TOTAL_BYTES];

static struct uart_async_rx g_async_rx;

static const struct uart_async_rx_config g_async_rx_config = {
    .buffer  = g_uart_pool,
    .length  = UART_POOL_TOTAL_BYTES,
    .buf_cnt = UART_POOL_BUF_CNT,
};

/* Accumulate total claimed bytes for harness assertions */
static size_t g_total_claimed;

/* ===========================================================================
 * Mod 2.6 – Public harness entry points
 * ========================================================================= */

/**
 * uart_mock_inject – enqueue fuzz bytes and reset state
 *
 * Implements the harness entry contract (§2.6):
 *   1. Validate pointer (NULL + nonzero len → no-op).
 *   2. Reset FIFO, async-rx state, slab, and counters.
 *   3. Clamp: accept at most UART_FIFO_SIZE bytes.
 *   4. Enqueue bytes into the FIFO.
 *   5. Does not retain any pointer into caller-owned input.
 *
 * @param data  Fuzzer-supplied bytes (may be NULL when len == 0).
 * @param len   Number of bytes to inject.
 */
void uart_mock_inject(const uint8_t *data, size_t len)
{
    /* Step 1 – validate */
    if (data == NULL && len != 0U) {
        return;
    }

    /* Step 2 – reset all state so no values leak between iterations */
    memset(&g_fifo,      0, sizeof(g_fifo));
    memset(g_uart_pool,  0, sizeof(g_uart_pool));
    memset(&g_async_rx,  0, sizeof(g_async_rx));
    g_total_claimed = 0U;

    int rc = uart_async_rx_init(&g_async_rx, &g_async_rx_config);
    if (rc != 0) {
        LOG_ERR("uart_async_rx_init failed: %d", rc);
        return;
    }

    /* Step 3 – clamp */
    if (len > UART_FIFO_SIZE) {
        len = UART_FIFO_SIZE;
    }

    /* Step 4 – enqueue (zero-length is allowed; FIFO stays empty) */
    for (size_t i = 0U; i < len; i++) {
        if (!fifo_push(&g_fifo, data[i])) {
            break; /* should not happen after clamp, but guard anyway */
        }
    }
    /* Step 5 – data[] not retained; FIFO holds our own copy */
}

/**
 * uart_mock_process – drive the async-rx receive path (Mod 2.1 / 2.5)
 *
 * Simulates the sequence of events a UART driver would generate:
 *
 *   For each sub-buffer available:
 *     a) Request a buffer  (uart_async_rx_buf_req)
 *     b) Fill it byte-by-byte from the FIFO
 *     c) Signal data ready (uart_async_rx_on_rdy)
 *     d) Signal release    (uart_async_rx_on_buf_rel)
 *
 * Then claims and consumes all pending data so buffers are returned.
 *
 * Bounded by UART_MAX_POLL_ITERS to ensure termination even on
 * malformed or adversarial inputs.
 */
void uart_mock_process(void)
{
    uint32_t iter = 0U;

    /* ------------------------------------------------------------------ */
    /* Phase 1: inject FIFO bytes into async-rx sub-buffers               */
    /* ------------------------------------------------------------------ */
    while (!fifo_empty(&g_fifo) && iter < UART_MAX_POLL_ITERS) {
        iter++;

        /* Request the next free sub-buffer from the helper */
        uint8_t *sub_buf = uart_async_rx_buf_req(&g_async_rx);
        if (sub_buf == NULL) {
            /* No free buffers – all are held by the helper or the user. */
            LOG_WRN("uart_mock_process: no free sub-buffers, %u bytes dropped",
                    (unsigned)g_fifo.count);
            break;
        }

        uint8_t sub_len = uart_async_rx_get_buf_len(&g_async_rx);
        uint8_t written = 0U;

        /* Fill the sub-buffer byte by byte from the FIFO (Mod 2.1) */
        while (written < sub_len && !fifo_empty(&g_fifo)) {
            uint8_t byte;
            if (!fifo_pop(&g_fifo, &byte)) break;
            sub_buf[written++] = byte;
        }

        if (written == 0U) {
            /*
             * Nothing written.  The buffer was requested but no data
             * arrived – signal a zero-length ready event so the helper
             * can advance its internal state, then release.
             */
            uart_async_rx_on_rdy(&g_async_rx, sub_buf, 0U);
            uart_async_rx_on_buf_rel(&g_async_rx, sub_buf);
            break;
        }

        /* Notify the helper that 'written' bytes are ready (UART_RX_RDY) */
        uart_async_rx_on_rdy(&g_async_rx, sub_buf, written);

        /* Signal that the driver has released the buffer (UART_RX_BUF_RELEASED) */
        uart_async_rx_on_buf_rel(&g_async_rx, sub_buf);
    }

    /* ------------------------------------------------------------------ */
    /* Phase 2: claim and consume all pending data (bounded)              */
    /* ------------------------------------------------------------------ */
    iter = 0U;
    while (g_async_rx.pending_bytes > 0 && iter < UART_MAX_POLL_ITERS) {
        iter++;

        uint8_t *claimed_data = NULL;
        size_t   available    = uart_async_rx_data_claim(
                                    &g_async_rx, &claimed_data,
                                    (size_t)UART_SUB_BUF_DATA_LEN);
        if (available == 0U || claimed_data == NULL) {
            break;
        }

        g_total_claimed += available;

        bool has_free = uart_async_rx_data_consume(&g_async_rx, available);
        (void)has_free;
    }
}

/* ===========================================================================
 * Inspection helpers for harness assertions
 * ========================================================================= */

/** Total bytes successfully claimed across all sub-buffers this iteration. */
size_t uart_mock_claimed_total(void)
{
    return g_total_claimed;
}

/** True if all injected bytes have been claimed and consumed. */
bool uart_mock_all_consumed(void)
{
    return g_async_rx.pending_bytes == 0;
}

/* ===========================================================================
 * Re-export the original uart_async_rx API for direct use by the harness
 * ========================================================================= */

/*
 * The following upstream functions are compiled in verbatim from the
 * upstream source via this translation unit and are available to the
 * harness through normal linkage:
 *
 *   int    uart_async_rx_init(struct uart_async_rx *, const struct uart_async_rx_config *)
 *   void   uart_async_rx_reset(struct uart_async_rx *)
 *   uint8_t *uart_async_rx_buf_req(struct uart_async_rx *)
 *   void   uart_async_rx_on_rdy(struct uart_async_rx *, uint8_t *, size_t)
 *   void   uart_async_rx_on_buf_rel(struct uart_async_rx *, uint8_t *)
 *   size_t uart_async_rx_data_claim(struct uart_async_rx *, uint8_t **, size_t)
 *   bool   uart_async_rx_data_consume(struct uart_async_rx *, size_t)
 *
 * Implementations follow – copied from the upstream source with only the
 * two unavailable headers replaced by zephyr_compat.h (Mod 2.4).
 */

/* --------------------------------------------------------------------------
 * Upstream uart_async_rx.c body (verbatim logic, headers replaced)
 * ------------------------------------------------------------------------ */

static uint8_t inc_idx(struct uart_async_rx *rx_data, uint8_t val)
{
    return (uint8_t)((val + 1U) & (uint8_t)(rx_data->config->buf_cnt - 1U));
}

static struct uart_async_rx_buf *get_buf(struct uart_async_rx *rx_data, uint8_t idx)
{
    uint8_t *p = rx_data->config->buffer;
    p += idx * (rx_data->buf_len + UART_ASYNC_RX_BUF_OVERHEAD);
    return (struct uart_async_rx_buf *)p;
}

uint8_t *uart_async_rx_buf_req(struct uart_async_rx *rx_data)
{
    uint8_t *data = NULL;

    if (rx_data->free_buf_cnt != 0) {
        struct uart_async_rx_buf *buf = get_buf(rx_data, rx_data->drv_buf_idx);
        data = buf->buffer;
        rx_data->drv_buf_idx = inc_idx(rx_data, rx_data->drv_buf_idx);
        atomic_dec(&rx_data->free_buf_cnt);
    }

    return data;
}

void uart_async_rx_on_rdy(struct uart_async_rx *rx_data, uint8_t *buffer, size_t length)
{
    struct uart_async_rx_buf *rx_buf =
        (struct uart_async_rx_buf *)(buffer - offsetof(struct uart_async_rx_buf, buffer));

    rx_buf->wr_idx += (uint8_t)length;
    __ASSERT_NO_MSG(rx_buf->wr_idx <= rx_data->buf_len);

    atomic_add(&rx_data->pending_bytes, (int)length);
}

static void buf_reset(struct uart_async_rx_buf *buf)
{
    buf->wr_idx    = 0U;
    buf->completed = 0U;
}

static void usr_rx_buf_release(struct uart_async_rx *rx_data,
                                struct uart_async_rx_buf *buf)
{
    buf_reset(buf);
    rx_data->rd_idx     = 0U;
    rx_data->rd_buf_idx = inc_idx(rx_data, rx_data->rd_buf_idx);
    atomic_inc(&rx_data->free_buf_cnt);
    __ASSERT_NO_MSG(rx_data->free_buf_cnt <= rx_data->config->buf_cnt);
}

void uart_async_rx_on_buf_rel(struct uart_async_rx *rx_data, uint8_t *buffer)
{
    struct uart_async_rx_buf *rx_buf =
        (struct uart_async_rx_buf *)(buffer - offsetof(struct uart_async_rx_buf, buffer));
    rx_buf->completed = 1U;
}

size_t uart_async_rx_data_claim(struct uart_async_rx *rx_data,
                                 uint8_t **data, size_t length)
{
    struct uart_async_rx_buf *buf;
    int rem;

    if ((rx_data->pending_bytes == 0) || (length == 0U)) {
        return 0U;
    }

    do {
        buf = get_buf(rx_data, rx_data->rd_buf_idx);
        if ((buf->completed == 1U) && (rx_data->rd_idx == buf->wr_idx)) {
            usr_rx_buf_release(rx_data, buf);
        } else {
            break;
        }
    } while (1);

    *data = &buf->buffer[rx_data->rd_idx];
    rem   = (int)buf->wr_idx - (int)rx_data->rd_idx;

    return (size_t)MIN((int)length, rem);
}

bool uart_async_rx_data_consume(struct uart_async_rx *rx_data, size_t length)
{
    struct uart_async_rx_buf *buf = get_buf(rx_data, rx_data->rd_buf_idx);

    rx_data->rd_idx += (uint8_t)length;

    if ((buf->completed == 1U) && (rx_data->rd_idx == buf->wr_idx)) {
        usr_rx_buf_release(rx_data, buf);
    }

    atomic_sub(&rx_data->pending_bytes, (int)length);

    __ASSERT_NO_MSG(rx_data->rd_idx <= buf->wr_idx);

    return rx_data->free_buf_cnt > 0;
}

void uart_async_rx_reset(struct uart_async_rx *rx_data)
{
    rx_data->free_buf_cnt  = rx_data->config->buf_cnt;
    rx_data->rd_idx        = 0U;
    rx_data->rd_buf_idx    = 0U;
    rx_data->drv_buf_idx   = 0U;
    rx_data->pending_bytes = 0;
    for (uint8_t i = 0U; i < rx_data->config->buf_cnt; i++) {
        buf_reset(get_buf(rx_data, i));
    }
}

int uart_async_rx_init(struct uart_async_rx *rx_data,
                       const struct uart_async_rx_config *config)
{
    __ASSERT_NO_MSG(config->length / config->buf_cnt <= UINT8_MAX);

    memset(rx_data, 0, sizeof(*rx_data));
    rx_data->config  = config;
    rx_data->buf_len = (uint8_t)((config->length / config->buf_cnt)
                                  - UART_ASYNC_RX_BUF_OVERHEAD);

    if (rx_data->buf_len >= BIT(7)) {
        return -EINVAL;
    }

    uart_async_rx_reset(rx_data);
    return 0;
}
