/*
 * zephyr_compat.h – Host-side compatibility shim for the Modbus RTU fuzzing target
 *
 * Provides every symbol required by modbus_rtu.c (refactored from
 * Zephyr's modbus_serial.c) so the file compiles without the Zephyr
 * build system.  Only definitions actually referenced by the
 * implementation are included; unused Zephyr APIs are intentionally
 * omitted.
 *
 * FUZZING ONLY – Do not ship in production firmware.
 */

#ifndef ZEPHYR_COMPAT_H
#define ZEPHYR_COMPAT_H

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
 * POSIX / standard error codes
 *
 * Zephyr uses negative errno values as return codes.  <errno.h> provides
 * the positive constants; the implementation returns them negated.
 * ========================================================================= */

/* errno.h already defines EINVAL, EIO, ENOMEM, ENODEV, etc.
 * Add any that may be absent on the host: */
#ifndef EMSGSIZE
#  define EMSGSIZE  90
#endif
#ifndef ENOTSUP
#  define ENOTSUP   95
#endif
#ifndef ETIMEDOUT
#  define ETIMEDOUT 110
#endif

/* =========================================================================
 * Logging macros
 *
 * Map to printf-based output as required by the prompt (§4.5).
 * LOG_DBG is silenced to avoid flooding AFL++ stdout.
 * ========================================================================= */

#define LOG_MODULE_REGISTER(name, level)   /* no-op */
#define LOG_MODULE_DECLARE(name, level)    /* no-op */

#define LOG_ERR(...) do { printf("[LOG_ERR] "); printf(__VA_ARGS__); printf("\n"); } while (0)
#define LOG_WRN(...) do { printf("[LOG_WRN] "); printf(__VA_ARGS__); printf("\n"); } while (0)
#define LOG_INF(...) do { printf("[LOG_INF] "); printf(__VA_ARGS__); printf("\n"); } while (0)
#define LOG_DBG(...) do { } while (0)

/* LOG_HEXDUMP_DBG – silenced in fuzzing builds */
#define LOG_HEXDUMP_DBG(buf, len, label)  do { } while (0)

/* =========================================================================
 * Modbus protocol constants
 * ========================================================================= */

/** Maximum RTU/ASCII frame byte count (Step 4.4 requirement) */
#define MODBUS_MAX_FRAME_SIZE      256U

/** Minimum RTU frame: 1 addr + 1 fc + 0 data + 2 CRC = 4 bytes */
#define MODBUS_RTU_MIN_MSG_SIZE    4U

/** CONFIG_MODBUS_BUFFER_SIZE – hard-coded to the 256-byte cap */
#define CONFIG_MODBUS_BUFFER_SIZE  MODBUS_MAX_FRAME_SIZE

/** Modbus operating modes */
#define MODBUS_MODE_RTU   0
#define MODBUS_MODE_ASCII 1

/* ASCII framing characters */
#define MODBUS_ASCII_START_FRAME_CHAR   ':'
#define MODBUS_ASCII_END_FRAME_CHAR1    '\r'
#define MODBUS_ASCII_END_FRAME_CHAR2    '\n'
#define MODBUS_ASCII_MIN_MSG_SIZE       3U

/* =========================================================================
 * Kernel timeout constants
 * ========================================================================= */

#define K_NO_WAIT    0
#define K_FOREVER   -1
#define K_USEC(us)  (us)
#define K_MSEC(ms)  ((ms) * 1000)

/* =========================================================================
 * Kernel synchronization primitives – single-threaded no-ops
 *
 * Return types match the Zephyr API so calls compile without warnings.
 * ========================================================================= */

typedef struct { int dummy; } k_sem;
typedef struct { int dummy; } k_mutex;
typedef struct { int dummy; } k_thread;

static inline void k_sem_init(k_sem *s, int initial, int limit)
    { (void)s; (void)initial; (void)limit; }
static inline int  k_sem_take(k_sem *s, int timeout)
    { (void)s; (void)timeout; return 0; }
static inline void k_sem_give(k_sem *s)
    { (void)s; }

static inline void k_mutex_init(k_mutex *m)
    { (void)m; }
static inline int  k_mutex_lock(k_mutex *m, int timeout)
    { (void)m; (void)timeout; return 0; }
static inline void k_mutex_unlock(k_mutex *m)
    { (void)m; }

/* =========================================================================
 * k_timer – stub that records user data so k_timer_user_data_get works
 * ========================================================================= */

typedef struct k_timer {
    void *user_data;
    void (*expiry_fn)(struct k_timer *);
    void (*stop_fn)(struct k_timer *);
} k_timer;

static inline void k_timer_init(k_timer *t,
                                 void (*expiry)(k_timer *),
                                 void (*stop)(k_timer *))
    { if (t) { t->expiry_fn = expiry; t->stop_fn = stop; t->user_data = NULL; } }
static inline void k_timer_start(k_timer *t, int dur, int period)
    { (void)t; (void)dur; (void)period; }
static inline void k_timer_stop(k_timer *t)
    { (void)t; }
static inline int  k_timer_status_get(k_timer *t)
    { (void)t; return 0; }
static inline void k_timer_user_data_set(k_timer *t, void *data)
    { if (t) t->user_data = data; }
static inline void *k_timer_user_data_get(k_timer *t)
    { return t ? t->user_data : NULL; }

/* =========================================================================
 * k_work / k_work_q – no-op stubs
 * ========================================================================= */

typedef struct k_work {
    void (*handler)(struct k_work *);
} k_work;

typedef struct { int dummy; } k_work_q;

static inline void k_work_init(k_work *w, void (*handler)(k_work *))
    { if (w) w->handler = handler; }
static inline int  k_work_submit(k_work *w)
    { (void)w; return 0; }

/* =========================================================================
 * Byte-order helpers
 *
 * sys_get_le16 / sys_put_le16 – little-endian (used by RTU CRC field).
 * sys_get_be16 / sys_put_be16 – big-endian (provided for completeness).
 * ========================================================================= */

static inline uint16_t sys_get_le16(const uint8_t *src)
{
    return (uint16_t)((uint16_t)src[0] | ((uint16_t)src[1] << 8));
}
static inline void sys_put_le16(uint16_t val, uint8_t *dst)
{
    dst[0] = (uint8_t)(val & 0xFFU);
    dst[1] = (uint8_t)(val >> 8);
}
static inline uint16_t sys_get_be16(const uint8_t *src)
{
    return (uint16_t)(((uint16_t)src[0] << 8) | (uint16_t)src[1]);
}
static inline void sys_put_be16(uint16_t val, uint8_t *dst)
{
    dst[0] = (uint8_t)(val >> 8);
    dst[1] = (uint8_t)(val & 0xFFU);
}

/* =========================================================================
 * CRC-16/ANSI (also called CRC-16/IBM, Modbus RTU CRC)
 *
 * Replaces <zephyr/sys/crc.h>: crc16_ansi()
 * NOTE: In the fuzzing target the CRC check is bypassed; this
 * implementation is kept for completeness and for the TX path.
 * ========================================================================= */

static inline uint16_t crc16_ansi(const uint8_t *data, size_t len)
{
    uint16_t crc = 0xFFFFU;
    for (size_t i = 0; i < len; i++) {
        crc ^= (uint16_t)data[i];
        for (int j = 0; j < 8; j++) {
            if (crc & 0x0001U) {
                crc = (crc >> 1) ^ 0xA001U;
            } else {
                crc >>= 1;
            }
        }
    }
    return crc;
}

/* =========================================================================
 * ASCII hex helpers  (replaces <zephyr/sys/util.h>: hex2bin, hex2char)
 * ========================================================================= */

static inline int hex2bin(const uint8_t *hex, size_t hexlen,
                           uint8_t *buf, size_t buflen)
{
    if (hexlen < 2 || buflen < 1) return 0;
    uint8_t hi = hex[0], lo = hex[1];
    hi = (hi >= 'a') ? (hi - 'a' + 10) :
         (hi >= 'A') ? (hi - 'A' + 10) : (hi - '0');
    lo = (lo >= 'a') ? (lo - 'a' + 10) :
         (lo >= 'A') ? (lo - 'A' + 10) : (lo - '0');
    buf[0] = (uint8_t)((hi << 4) | lo);
    return 1;
}

static inline void hex2char(uint8_t nibble, uint8_t *c)
{
    *c = (nibble < 10) ? ('0' + nibble) : ('A' + nibble - 10);
}

/* =========================================================================
 * GPIO stub types (used by modbus_serial_config for DE/RE pins)
 * ========================================================================= */

struct gpio_dt_spec {
    const void *port;   /* unused – always NULL in stubs */
    uint32_t    pin;
    uint32_t    dt_flags;
};

#define GPIO_OUTPUT_INACTIVE  0U

static inline bool device_is_ready(const void *dev) { (void)dev; return true; }
static inline int  gpio_pin_set(const void *port, uint32_t pin, int val)
    { (void)port; (void)pin; (void)val; return 0; }
static inline int  gpio_pin_configure_dt(const struct gpio_dt_spec *spec, uint32_t flags)
    { (void)spec; (void)flags; return 0; }

/* =========================================================================
 * UART stub types (used by modbus_serial_config)
 * ========================================================================= */

struct device { const char *name; };

#define UART_CFG_FLOW_CTRL_NONE  0
#define UART_CFG_DATA_BITS_7     7
#define UART_CFG_DATA_BITS_8     8
#define UART_CFG_PARITY_NONE     0
#define UART_CFG_PARITY_ODD      1
#define UART_CFG_PARITY_EVEN     2
#define UART_CFG_STOP_BITS_0_5   0
#define UART_CFG_STOP_BITS_1     1
#define UART_CFG_STOP_BITS_1_5   2
#define UART_CFG_STOP_BITS_2     3

struct uart_config {
    uint32_t baudrate;
    uint8_t  flow_ctrl;
    uint8_t  data_bits;
    uint8_t  parity;
    uint8_t  stop_bits;
};

static inline int  uart_configure(const struct device *dev,
                                   const struct uart_config *cfg)
    { (void)dev; (void)cfg; return 0; }
static inline void uart_irq_rx_enable(const struct device *dev)  { (void)dev; }
static inline void uart_irq_rx_disable(const struct device *dev) { (void)dev; }
static inline void uart_irq_tx_enable(const struct device *dev)  { (void)dev; }
static inline void uart_irq_tx_disable(const struct device *dev) { (void)dev; }
static inline int  uart_irq_update(const struct device *dev)     { (void)dev; return 0; }
static inline int  uart_irq_is_pending(const struct device *dev) { (void)dev; return 0; }
static inline int  uart_irq_rx_ready(const struct device *dev)   { (void)dev; return 0; }
static inline int  uart_irq_tx_ready(const struct device *dev)   { (void)dev; return 0; }
static inline int  uart_irq_tx_complete(const struct device *dev){ (void)dev; return 1; }
static inline int  uart_fifo_read(const struct device *dev,
                                   uint8_t *buf, int size)
    { (void)dev; (void)buf; (void)size; return 0; }
static inline int  uart_fifo_fill(const struct device *dev,
                                   const uint8_t *buf, int size)
    { (void)dev; (void)buf; (void)size; return size; }
static inline void uart_irq_callback_user_data_set(const struct device *dev,
                                                    void *cb, void *data)
    { (void)dev; (void)cb; (void)data; }

/* IS_ENABLED – always false for optional Zephyr Kconfig symbols */
#define IS_ENABLED(x)              0
#define CONFIG_UART_USE_RUNTIME_CONFIGURE  0

/* =========================================================================
 * Modbus ADU (Application Data Unit) structures
 * ========================================================================= */

struct modbus_adu {
    uint8_t  unit_id;                              /* slave address       */
    uint8_t  fc;                                   /* function code       */
    uint16_t length;                               /* payload byte count  */
    uint16_t crc;                                  /* RTU CRC field       */
    uint8_t  data[MODBUS_MAX_FRAME_SIZE];          /* PDU payload         */
};

/* =========================================================================
 * modbus_serial_config – mirrors Zephyr's internal serial driver config
 * ========================================================================= */

struct modbus_serial_config {
    /* Hardware device stubs */
    const struct device        *dev;    /* UART device – NULL in fuzzing   */
    const struct gpio_dt_spec  *de;     /* Driver-Enable GPIO – NULL       */
    const struct gpio_dt_spec  *re;     /* Receiver-Enable GPIO – NULL     */

    /* RTU inter-frame silence timer */
    k_timer  rtu_timer;
    uint32_t rtu_timeout;               /* microseconds                    */

    /* Byte-level serial buffer (shared RX/TX) */
    uint8_t  uart_buf[MODBUS_MAX_FRAME_SIZE];
    uint8_t *uart_buf_ptr;              /* write cursor                    */
    uint16_t uart_buf_ctr;              /* bytes currently in buffer       */
};

/* =========================================================================
 * modbus_iface_param – interface initialisation parameters
 * ========================================================================= */

struct modbus_serial_param {
    uint32_t baud;
    uint8_t  parity;
    uint8_t  stop_bits_client;
};

struct modbus_iface_param {
    uint8_t                    mode;    /* MODBUS_MODE_RTU / ASCII         */
    struct modbus_serial_param serial;
};

/* =========================================================================
 * modbus_context – top-level driver context
 * ========================================================================= */

struct modbus_context {
    /* Pointer to serial-layer config (cast from driver data) */
    struct modbus_serial_config *cfg;

    /* Operating mode */
    uint8_t  mode;      /* MODBUS_MODE_RTU or MODBUS_MODE_ASCII            */
    bool     client;    /* true → client role; false → server role          */

    /* Received and transmit ADUs */
    struct modbus_adu rx_adu;
    struct modbus_adu tx_adu;

    /* Work item for deferred frame processing (no-op in fuzzing) */
    k_work server_work;
};

/* =========================================================================
 * Miscellaneous helpers
 * ========================================================================= */

#define ARRAY_SIZE(arr)  (sizeof(arr) / sizeof((arr)[0]))
#define BIT(n)           (1U << (n))

#include <assert.h>
#define __ASSERT(cond, fmt, ...)  assert(cond)
#define __ASSERT_NO_MSG(cond)     assert(cond)

#endif /* ZEPHYR_COMPAT_H */
