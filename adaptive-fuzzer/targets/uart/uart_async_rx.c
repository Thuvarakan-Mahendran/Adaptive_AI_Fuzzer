/*
 * uart_async_rx.c – Zephyr async UART RX driver adapted for fuzzing
 *
 * Source: Adapted from Zephyr RTOS drivers/serial/uart_async_rx.c
 *         (Apache-2.0 / Zephyr Project)
 *
 * Key adaptations (per paper Section III-D):
 *   1. All hardware ISR registration (IRQ_CONNECT, irq_enable/disable) removed.
 *   2. Hardware UART RX register polling replaced by a software ring-buffer FIFO.
 *   3. DMA channel configuration and buffer-swapping replaced by direct memcpy
 *      from the software FIFO into two statically allocated reception pools.
 *   4. Busy-wait polling loops replaced by counter-bounded FIFO checks.
 *   5. Public API: uart_async_rx_reset(), uart_async_rx_push(), uart_async_rx_process().
 */

#include "include/zephyr_compat.h"
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>

/* -------------------------------------------------------------------------
 * Configuration
 * ---------------------------------------------------------------------- */

/** Maximum bytes in the software FIFO */
#define UART_FIFO_SIZE      512u

/** Maximum bytes in each of the two double-buffered reception pools */
#define UART_POOL_BUF_SIZE  256u

/** Number of reception pool buffers (double-buffering) */
#define UART_POOL_COUNT     2u

/** Maximum loop iterations for poll-based reads (prevents fuzzer hangs) */
#define UART_MAX_POLL_ITERS 4096u

/* -------------------------------------------------------------------------
 * Software ring-buffer FIFO (replaces hardware UART RX FIFO, Step 3.3)
 * ---------------------------------------------------------------------- */

typedef struct {
    uint8_t  buf[UART_FIFO_SIZE];
    uint32_t head; /* write index */
    uint32_t tail; /* read index  */
    uint32_t count;
} sw_fifo_t;

static sw_fifo_t g_fifo;

static inline bool fifo_empty(const sw_fifo_t *f) { return f->count == 0; }
static inline bool fifo_full(const sw_fifo_t *f)  { return f->count == UART_FIFO_SIZE; }

static bool fifo_push(sw_fifo_t *f, uint8_t byte)
{
    if (fifo_full(f)) return false;
    f->buf[f->head] = byte;
    f->head = (f->head + 1) % UART_FIFO_SIZE;
    f->count++;
    return true;
}

static bool fifo_pop(sw_fifo_t *f, uint8_t *out)
{
    if (fifo_empty(f)) return false;
    *out = f->buf[f->tail];
    f->tail = (f->tail + 1) % UART_FIFO_SIZE;
    f->count--;
    return true;
}

/* -------------------------------------------------------------------------
 * Static double-buffered reception pools (replaces DMA buffers, Step 3.4)
 * ---------------------------------------------------------------------- */

typedef struct {
    uint8_t  data[UART_POOL_BUF_SIZE];
    uint32_t used;   /* bytes written into this buffer */
    bool     active; /* true when this buffer is the current write target */
} rx_pool_buf_t;

static rx_pool_buf_t g_pool[UART_POOL_COUNT];
static uint32_t      g_active_buf; /* index of the currently active pool */
static uint32_t      g_rx_total;   /* total bytes received this iteration   */

/* -------------------------------------------------------------------------
 * Receiver state machine
 * ---------------------------------------------------------------------- */

typedef enum {
    UART_RX_STATE_IDLE = 0,
    UART_RX_STATE_RECEIVING,
    UART_RX_STATE_DONE,
    UART_RX_STATE_ERROR,
} uart_rx_state_t;

static uart_rx_state_t g_rx_state;

/* Mock Zephyr device – declared here; also referenced by zephyr_compat.h */
struct device mock_uart_device = { .name = "UART_0", .data = NULL,
                                   .config = NULL, .api = NULL };

/* -------------------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------------- */

/*
 * uart_async_rx_reset – reset all driver state
 *
 * Called by the harness between fuzz iterations (Step 3.6).
 */
void uart_async_rx_reset(void)
{
    memset(&g_fifo,  0, sizeof(g_fifo));
    memset(&g_pool,  0, sizeof(g_pool));
    g_active_buf = 0;
    g_rx_total   = 0;
    g_rx_state   = UART_RX_STATE_IDLE;
    g_pool[0].active = true;
}

/*
 * uart_async_rx_push – inject fuzzer bytes into the software FIFO
 *
 * Simulates bytes arriving on the UART RX line (from the peripheral stream).
 * Returns the number of bytes successfully enqueued.
 */
size_t uart_async_rx_push(const uint8_t *data, size_t len)
{
    size_t pushed = 0;
    for (size_t i = 0; i < len; i++) {
        if (!fifo_push(&g_fifo, data[i])) break;
        pushed++;
    }
    return pushed;
}

/*
 * uart_async_rx_process – transfer FIFO bytes into the static pool
 *
 * Replaces the ISR-driven / DMA-driven transfer path.  Reads bytes from
 * the software FIFO (counter-bounded to prevent hang) and copies them into
 * the active reception pool, swapping buffers when one is full.
 *
 * Returns total bytes processed in this call.
 */
size_t uart_async_rx_process(void)
{
    size_t   processed  = 0;
    uint32_t iter_guard = 0;

    g_rx_state = UART_RX_STATE_RECEIVING;

    while (!fifo_empty(&g_fifo) && iter_guard < UART_MAX_POLL_ITERS) {
        iter_guard++;

        rx_pool_buf_t *active = &g_pool[g_active_buf];

        /* If the active pool is full, swap to the other buffer (double-buffering) */
        if (active->used >= UART_POOL_BUF_SIZE) {
            active->active = false;

            uint32_t next_buf = (g_active_buf + 1) % UART_POOL_COUNT;
            /* Consume / reset the next buffer before re-use */
            g_pool[next_buf].used   = 0;
            g_pool[next_buf].active = true;
            g_active_buf            = next_buf;
            active                  = &g_pool[g_active_buf];

            LOG_DBG("Buffer swap → active pool %u", g_active_buf);
        }

        /* Pull one byte from FIFO and copy into active pool (replaces DMA, Step 3.4) */
        uint8_t byte;
        if (!fifo_pop(&g_fifo, &byte)) break;

        active->data[active->used] = byte;
        active->used++;
        g_rx_total++;
        processed++;
    }

    g_rx_state = UART_RX_STATE_DONE;
    LOG_DBG("uart_async_rx_process: processed=%zu total=%u", processed, g_rx_total);
    return processed;
}

/*
 * uart_async_rx_get_pool – expose the pool for assertion checks in harness
 *
 * Returns a pointer to pool buffer @idx and writes its used-byte count to
 * *out_used.
 */
const uint8_t *uart_async_rx_get_pool(uint32_t idx, uint32_t *out_used)
{
    if (idx >= UART_POOL_COUNT) return NULL;
    if (out_used) *out_used = g_pool[idx].used;
    return g_pool[idx].data;
}

uint32_t uart_async_rx_total(void) { return g_rx_total; }
