/*
 * zephyr_compat.h - Mock Zephyr RTOS dependencies for fuzzing
 *
 * Replaces Zephyr-specific types, logging macros, and kernel synchronization
 * primitives with standard C equivalents so modbus_rtu.c can compile and run
 * on a host without the Zephyr build system.
 */

#ifndef ZEPHYR_COMPAT_H
#define ZEPHYR_COMPAT_H

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* ---------------------------------------------------------------------------
 * Standard integer type aliases (already provided by <stdint.h>; listed here
 * for documentation purposes in case any Zephyr-specific aliases are needed).
 * --------------------------------------------------------------------------- */

/* Zephyr uses these exact names, which are already defined by <stdint.h>:
 *   uint8_t, uint16_t, uint32_t, int8_t, int16_t, int32_t, bool
 */

/* ---------------------------------------------------------------------------
 * Logging macro stubs
 *
 * Replace Zephyr's structured logging with printf-based stubs.  In fuzzing
 * builds, noisy logging can mask coverage signals, so these are intentionally
 * quiet (compiled-out) by default.  Define FUZZER_VERBOSE to enable them.
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

/* LOG_MODULE_REGISTER / LOG_MODULE_DECLARE are no-ops outside Zephyr. */
#define LOG_MODULE_REGISTER(name, level)  /* no-op */
#define LOG_MODULE_DECLARE(name, level)   /* no-op */

/* ---------------------------------------------------------------------------
 * Kernel synchronization primitive stubs
 *
 * These types are used by Modbus code to guard shared state.  Under the fuzzer
 * we run single-threaded, so all operations become no-ops.
 * --------------------------------------------------------------------------- */

typedef struct { int dummy; } k_sem;
typedef struct { int dummy; } k_mutex;
typedef struct { int dummy; } k_timer;
typedef struct { int dummy; } k_thread;
typedef struct { int dummy; } k_work;
typedef struct { int dummy; } k_work_q;

#define k_sem_init(sem, initial, limit)    do {} while (0)
#define k_sem_take(sem, timeout)           (0)
#define k_sem_give(sem)                    do {} while (0)
#define k_mutex_init(m)                    do {} while (0)
#define k_mutex_lock(m, timeout)           (0)
#define k_mutex_unlock(m)                  do {} while (0)
#define k_timer_init(t, fn, stop_fn)       do {} while (0)
#define k_timer_start(t, dur, period)      do {} while (0)
#define k_timer_stop(t)                    do {} while (0)
#define k_timer_status_get(t)              (0)

/* Zephyr timeout constants */
#define K_NO_WAIT     0
#define K_FOREVER    -1
#define K_MSEC(ms)   (ms)

/* ---------------------------------------------------------------------------
 * Miscellaneous Zephyr macros
 * --------------------------------------------------------------------------- */

#define ARRAY_SIZE(arr)  (sizeof(arr) / sizeof((arr)[0]))
#define BIT(n)           (1U << (n))

/* __ASSERT / __ASSERT_NO_MSG – map to standard assert */
#include <assert.h>
#define __ASSERT(cond, fmt, ...)   assert(cond)
#define __ASSERT_NO_MSG(cond)      assert(cond)

/* Zephyr sys_put_be16 / sys_get_be16 helpers */
static inline void sys_put_be16(uint16_t val, uint8_t *dst) {
    dst[0] = (uint8_t)(val >> 8);
    dst[1] = (uint8_t)(val & 0xFF);
}
static inline uint16_t sys_get_be16(const uint8_t *src) {
    return (uint16_t)((src[0] << 8) | src[1]);
}

#endif /* ZEPHYR_COMPAT_H */
