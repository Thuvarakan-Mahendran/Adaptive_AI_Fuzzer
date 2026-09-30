/*
 * zephyr_compat.h - Mock Zephyr RTOS dependencies for UART async-rx fuzzing
 *
 * Provides host-side stubs for Zephyr types, device-tree bindings, interrupt
 * registration primitives, and DMA structures so that uart_async_rx.c can be
 * compiled and exercised on a native x86-64 host.
 */

#ifndef ZEPHYR_COMPAT_UART_H
#define ZEPHYR_COMPAT_UART_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

/* ---------------------------------------------------------------------------
 * Primitive type aliases
 * --------------------------------------------------------------------------- */

typedef int32_t  k_timeout_t;
typedef uint32_t k_ticks_t;

/* ---------------------------------------------------------------------------
 * Logging stubs
 * --------------------------------------------------------------------------- */

#ifdef FUZZER_VERBOSE
#  define LOG_ERR(fmt, ...)  fprintf(stderr, "[ERR] " fmt "\n", ##__VA_ARGS__)
#  define LOG_WRN(fmt, ...)  fprintf(stderr, "[WRN] " fmt "\n", ##__VA_ARGS__)
#  define LOG_INF(fmt, ...)  fprintf(stderr, "[INF] " fmt "\n", ##__VA_ARGS__)
#  define LOG_DBG(fmt, ...)  fprintf(stderr, "[DBG] " fmt "\n", ##__VA_ARGS__)
#else
#  define LOG_ERR(fmt, ...)  do {} while (0)
#  define LOG_WRN(fmt, ...)  do {} while (0)
#  define LOG_INF(fmt, ...)  do {} while (0)
#  define LOG_DBG(fmt, ...)  do {} while (0)
#endif

#define LOG_MODULE_REGISTER(name, level)  /* no-op */
#define LOG_MODULE_DECLARE(name, level)   /* no-op */

/* ---------------------------------------------------------------------------
 * Mock hardware device / device-tree structures
 *
 * Zephyr passes a `const struct device *dev` to driver callbacks.  We expose
 * a single static mock device that the harness passes everywhere.
 * --------------------------------------------------------------------------- */

struct device {
    const char *name;
    void       *data;   /* driver-private data pointer */
    void       *config; /* driver config pointer */
    void       *api;    /* driver API pointer */
};

/* Macro used in Zephyr drivers: DEVICE_DT_GET(node_id) → static mock device */
extern struct device mock_uart_device;
#define DEVICE_DT_GET(node)  (&mock_uart_device)
#define DEVICE_DT_INST_GET(inst) (&mock_uart_device)

/* ---------------------------------------------------------------------------
 * Interrupt stubs (IRQ_CONNECT, irq_enable, irq_disable)
 * --------------------------------------------------------------------------- */

#define IRQ_CONNECT(irq, pri, fn, arg, flags)  do {} while (0)
#define irq_enable(irq)                         do {} while (0)
#define irq_disable(irq)                        do {} while (0)

/* ---------------------------------------------------------------------------
 * Kernel synchronization stubs (single-threaded fuzzer → all no-ops)
 * --------------------------------------------------------------------------- */

typedef struct { int dummy; } k_sem;
typedef struct { int dummy; } k_mutex;
typedef struct { int dummy; } k_work;
typedef struct { int dummy; } k_work_q;

#define k_sem_init(s, init, lim)     do {} while (0)
#define k_sem_take(s, t)             (0)
#define k_sem_give(s)                do {} while (0)
#define k_mutex_init(m)              do {} while (0)
#define k_mutex_lock(m, t)           (0)
#define k_mutex_unlock(m)            do {} while (0)

#define K_NO_WAIT   0
#define K_FOREVER  -1
#define K_MSEC(ms) (ms)

/* ---------------------------------------------------------------------------
 * DMA / scatter-gather stubs
 * --------------------------------------------------------------------------- */

struct dma_block_config {
    uint32_t source_address;
    uint32_t dest_address;
    uint32_t block_size;
    struct dma_block_config *next_block;
};

struct dma_config {
    uint32_t channel_direction;
    uint32_t source_data_size;
    uint32_t dest_data_size;
    uint32_t source_burst_length;
    uint32_t dest_burst_length;
    uint32_t block_count;
    struct dma_block_config *head_block;
    void (*dma_callback)(const struct device *dev, void *user_data,
                         uint32_t channel, int status);
    void *user_data;
};

#define DMA_MEMORY_TO_MEMORY  0
#define DMA_PERIPHERAL_TO_MEMORY 1

/* DMA API stubs – always succeed without doing hardware I/O */
static inline int dma_config(const struct device *dev, uint32_t channel,
                              struct dma_config *config) {
    (void)dev; (void)channel; (void)config;
    return 0;
}
static inline int dma_start(const struct device *dev, uint32_t channel) {
    (void)dev; (void)channel;
    return 0;
}
static inline int dma_stop(const struct device *dev, uint32_t channel) {
    (void)dev; (void)channel;
    return 0;
}

/* ---------------------------------------------------------------------------
 * Miscellaneous Zephyr helpers
 * --------------------------------------------------------------------------- */

#define ARRAY_SIZE(arr)   (sizeof(arr) / sizeof((arr)[0]))
#define BIT(n)            (1U << (n))
#define CONTAINER_OF(ptr, type, field) \
    ((type *)((char *)(ptr) - offsetof(type, field)))

#define __ASSERT(cond, fmt, ...)  assert(cond)
#define __ASSERT_NO_MSG(cond)     assert(cond)

#endif /* ZEPHYR_COMPAT_UART_H */
