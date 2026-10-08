/*
 * operators.c – C mutation operator pool implementation.
 *
 * Strategy (what to change) is completely decoupled from offset selection
 * (where in the buffer to change it).  rand() is used for offsets and
 * interesting-value indices; the PPO agent is responsible only for choosing
 * which operator to invoke.
 *
 * All operators respect the max_size hard cap for growing operations
 * (BLOCK_INS, SPLICE) and safely handle edge cases (size == 0, size == 1).
 */

#include "operators.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* -------------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------------- */

/* Safe random offset within [0, size) */
static inline size_t rand_off(size_t size)
{
    return (size > 1) ? (size_t)rand() % size : 0;
}

/* XOR a window of bytes with mask, clamped to buffer bounds */
static inline void xor_window(uint8_t *buf, size_t off, size_t len,
                               size_t size, uint8_t mask)
{
    for (size_t i = off; i < off + len && i < size; i++)
        buf[i] ^= mask;
}

/* Flip `nbits` consecutive bits starting at byte `off`, bit `bit_start` */
static inline void flip_bits(uint8_t *buf, size_t size, int nbits)
{
    if (size == 0) return;
    size_t byte_off  = rand_off(size);
    int    bit_start = rand() % 8;
    for (int b = 0; b < nbits; b++) {
        size_t bi = byte_off + (size_t)((bit_start + b) / 8);
        if (bi >= size) break;
        buf[bi] ^= (uint8_t)(1u << ((bit_start + b) % 8));
    }
}

/* -------------------------------------------------------------------------
 * mutate_apply
 * ------------------------------------------------------------------------- */
size_t mutate_apply(uint16_t op, uint8_t *buf, size_t size,
                    const uint8_t *add_buf, size_t add_size,
                    size_t max_size)
{
    if (!buf || size == 0)
        return 0;

    size_t off;

    switch ((mutator_op_t)op) {

    /* ---- Bit flips --------------------------------------------------- */
    case OP_BITFLIP_1_1:
        flip_bits(buf, size, 1);
        return size;

    case OP_BITFLIP_2_1:
        flip_bits(buf, size, 2);
        return size;

    case OP_BITFLIP_4_1:
        flip_bits(buf, size, 4);
        return size;

    /* ---- Byte flips -------------------------------------------------- */
    case OP_BYTEFLIP_1:
        buf[rand_off(size)] ^= 0xFF;
        return size;

    case OP_BYTEFLIP_2:
        off = rand_off(size);
        xor_window(buf, off, 2, size, 0xFF);
        return size;

    case OP_BYTEFLIP_4:
        off = rand_off(size);
        xor_window(buf, off, 4, size, 0xFF);
        return size;

    /* ---- Arithmetic -------------------------------------------------- */
    case OP_ARITH_INC_8:
        buf[rand_off(size)] += 1;
        return size;

    case OP_ARITH_DEC_8:
        buf[rand_off(size)] -= 1;
        return size;

    case OP_ARITH_INC_16:
        if (size >= 2) {
            /* Pick a random aligned uint16 offset */
            size_t word_off = (rand_off(size - 1)) & ~(size_t)1;
            uint16_t val;
            memcpy(&val, buf + word_off, 2);
            val += 1;
            memcpy(buf + word_off, &val, 2);
        }
        return size;

    case OP_ARITH_DEC_16:
        if (size >= 2) {
            size_t word_off = (rand_off(size - 1)) & ~(size_t)1;
            uint16_t val;
            memcpy(&val, buf + word_off, 2);
            val -= 1;
            memcpy(buf + word_off, &val, 2);
        }
        return size;

    /* ---- Interesting values ------------------------------------------ */
    case OP_INTERESTING_8:
        buf[rand_off(size)] =
            (uint8_t)MUT_INTERESTING_8[rand() % MUT_INTERESTING_8_LEN];
        return size;

    /* ---- Block operations -------------------------------------------- */
    case OP_BLOCK_DEL:
        if (size > 2) {
            off = rand_off(size - 1);                     /* delete start */
            size_t del_len = 1 + (size_t)rand() % (size - off);
            if (off + del_len > size) del_len = size - off;
            memmove(buf + off, buf + off + del_len, size - (off + del_len));
            return size - del_len;
        }
        return size;

    case OP_BLOCK_INS:
        if (size < max_size) {
            off = rand_off(size);
            memmove(buf + off + 1, buf + off, size - off);
            buf[off] = (uint8_t)rand();
            return size + 1;
        }
        return size;

    /* ---- Splice ------------------------------------------------------- */
    case OP_SPLICE:
        if (add_buf && add_size > 0 && size < max_size) {
            size_t room     = max_size - size;
            size_t copy_len = (add_size < room) ? add_size : room;
            memcpy(buf + size, add_buf, copy_len);
            return size + copy_len;
        }
        /* Fallback when no secondary seed: random XOR at random offset */
        buf[rand_off(size)] ^= 0x55;
        return size;

    /* ---- Unknown operator -------------------------------------------- */
    default:
        /* Safe no-op fallback: single-byte XOR */
        buf[rand_off(size)] ^= 0x55;
        return size;
    }
}
