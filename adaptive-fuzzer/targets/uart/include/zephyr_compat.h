/*
 * zephyr_compat.h – Host-side compatibility shim for the UART async-rx fuzzing target
 *
 * Provides every symbol required by the refactored uart_async_rx.c so it
 * compiles without the Zephyr build system on a standard C11 host.
 *
 * What the upstream actually uses (from inspection of
 *   zephyr-upstream/drivers/serial/uart_async_rx.c  and
 *   zephyr-upstream/include/zephyr/drivers/serial/uart_async_rx.h):
 *
 *   - <string.h>                   → memset, offsetof
 *   - <zephyr/drivers/uart.h>      → not directly called; pulled in for types
 *   - <zephyr/drivers/serial/uart_async_rx.h>
 *       struct uart_async_rx_buf   (wr_idx:7, completed:1, buffer[])
 *       struct uart_async_rx       (config, pending_bytes, free_buf_cnt,
 *                                   buf_len, drv_buf_idx, rd_buf_idx, rd_idx)
 *       struct uart_async_rx_config (buffer, length, buf_cnt)
 *       UART_ASYNC_RX_BUF_OVERHEAD
 *   - atomic_t, atomic_dec/inc/add/sub
 *   - BIT(n), MIN(a,b)
 *   - __ASSERT_NO_MSG
 *   - EINVAL
 *
 * FUZZING ONLY – Do not ship in production firmware.
 */

#ifndef ZEPHYR_COMPAT_UART_H
#define ZEPHYR_COMPAT_UART_H

/* =========================================================================
 * Standard C headers
 * ========================================================================= */

#include <assert.h>
#include <errno.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Logging macros (mod 2.3)
 *
 * LOG_ERR / LOG_WRN / LOG_INF → printf / fprintf(stderr)
 * LOG_DBG → silenced to reduce fuzzing overhead.
 *
 * The upstream uart_async_rx.c does not call any LOG_* macros itself,
 * but the harness and any future extension may, so definitions are
 * provided here.  The variadic form works even with zero extra arguments
 * because the trailing ##__VA_ARGS__ token-paste suppresses the comma.
 * ========================================================================= */

#define LOG_MODULE_REGISTER(name, level)   /* no-op */
#define LOG_MODULE_DECLARE(name, level)    /* no-op */

#define LOG_ERR(fmt, ...) \
    do { fprintf(stderr, "[LOG_ERR] " fmt "\n", ##__VA_ARGS__); } while (0)
#define LOG_WRN(fmt, ...) \
    do { fprintf(stderr, "[LOG_WRN] " fmt "\n", ##__VA_ARGS__); } while (0)
#define LOG_INF(fmt, ...) \
    do { printf("[LOG_INF] " fmt "\n", ##__VA_ARGS__); } while (0)
#define LOG_DBG(fmt, ...) \
    do { } while (0)

/* =========================================================================
 * Assertions
 * ========================================================================= */

#define __ASSERT_NO_MSG(cond)      assert(cond)
#define __ASSERT(cond, fmt, ...)   assert(cond)

/* =========================================================================
 * atomic_t – single-threaded fuzzer: plain int, no atomics needed
 *
 * Zephyr's atomic operations (atomic_inc, atomic_dec, atomic_add,
 * atomic_sub) modify an atomic_t by value and return the previous value.
 * Here we use plain int with cast stubs; thread-safety is irrelevant.
 * ========================================================================= */

typedef int atomic_t;

static inline int atomic_inc(atomic_t *target)
{
    int old = *target;
    (*target)++;
    return old;
}

static inline int atomic_dec(atomic_t *target)
{
    int old = *target;
    (*target)--;
    return old;
}

static inline int atomic_add(atomic_t *target, int val)
{
    int old = *target;
    *target += val;
    return old;
}

static inline int atomic_sub(atomic_t *target, int val)
{
    int old = *target;
    *target -= val;
    return old;
}

/* =========================================================================
 * Utility macros used by the upstream source
 * ========================================================================= */

#ifndef MIN
#  define MIN(a, b)  ((a) < (b) ? (a) : (b))
#endif

#ifndef BIT
#  define BIT(n)  (1U << (n))
#endif

/* =========================================================================
 * UART async-rx structures and constants
 *
 * Copied verbatim from:
 *   zephyr-upstream/include/zephyr/drivers/serial/uart_async_rx.h
 *
 * These are the types actually used by uart_async_rx.c.  They are placed
 * here so the refactored standalone source needs only one include path.
 * ========================================================================= */

/**
 * @brief RX buffer structure which holds the buffer and its state.
 *
 * The buffer[] flexible array member is allocated as part of a larger
 * slab; see get_buf() in uart_async_rx.c.
 */
struct uart_async_rx_buf {
    /* Write index incremented when new data is reported received. */
    uint8_t wr_idx    : 7;
    /* Set to 1 when the buffer is released by the driver. */
    uint8_t completed : 1;
    /* Data location passed to the UART driver. */
    uint8_t buffer[];
};

/** @brief Amount of per-buffer management overhead in bytes. */
#define UART_ASYNC_RX_BUF_OVERHEAD  offsetof(struct uart_async_rx_buf, buffer)

/** @brief UART asynchronous RX helper configuration. */
struct uart_async_rx_config {
    uint8_t *buffer;   /**< Pointer to the backing buffer slab. */
    size_t   length;   /**< Total length of the slab in bytes.  */
    uint8_t  buf_cnt;  /**< Number of sub-buffers in the slab.  */
};

/** @brief UART asynchronous RX helper instance state. */
struct uart_async_rx {
    const struct uart_async_rx_config *config;

    /** Total pending bytes across all sub-buffers. */
    atomic_t pending_bytes;

    /** Number of sub-buffers currently free (not held by driver). */
    atomic_t free_buf_cnt;

    /** Single sub-buffer data capacity (bytes, without overhead). */
    uint8_t  buf_len;

    /** Index of the next sub-buffer to hand to the UART driver. */
    uint8_t  drv_buf_idx;

    /** Index of the sub-buffer currently being consumed by the user. */
    uint8_t  rd_buf_idx;

    /** Read cursor within the current consume buffer. */
    uint8_t  rd_idx;
};

/* =========================================================================
 * Inline accessor – matches the static inline in uart_async_rx.h
 * ========================================================================= */

static inline uint8_t uart_async_rx_get_buf_len(struct uart_async_rx *async_rx)
{
    return async_rx->buf_len;
}

/* =========================================================================
 * Miscellaneous helpers
 * ========================================================================= */

#define ARRAY_SIZE(arr)  (sizeof(arr) / sizeof((arr)[0]))

/* CONTAINER_OF – needed by harness, not by the upstream source itself */
#define CONTAINER_OF(ptr, type, field) \
    ((type *)((char *)(ptr) - offsetof(type, field)))

/* Timeout constants – not used by uart_async_rx.c but kept for completeness */
typedef int32_t k_timeout_t;
#define K_NO_WAIT   0
#define K_FOREVER  -1

#endif /* ZEPHYR_COMPAT_UART_H */
