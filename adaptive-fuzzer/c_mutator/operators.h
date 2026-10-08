/*
 * operators.h – Frozen C mutation operator pool.
 *
 * The 14 operators below define the discrete action space for the PPO
 * scheduler.  Their integer indices must remain stable between builds
 * because the Python PPO model is trained with these IDs.
 *
 * ACTION_DIM in ppo_scheduler/config.py must be set to TOTAL_OPERATORS (14)
 * or any subset ≤ 14 when integrating.
 */

#ifndef OPERATORS_H
#define OPERATORS_H

#include <stdint.h>
#include <stddef.h>

/* -------------------------------------------------------------------------
 * Frozen operator enumeration (PPO action space)
 * Do NOT reorder or renumber — the Python PPO model is trained on these IDs.
 * ------------------------------------------------------------------------- */
typedef enum {
    OP_BITFLIP_1_1   =  0,  /* Flip 1 bit at a random byte offset            */
    OP_BITFLIP_2_1   =  1,  /* Flip 2 consecutive bits                       */
    OP_BITFLIP_4_1   =  2,  /* Flip 4 consecutive bits (nybble)              */
    OP_BYTEFLIP_1    =  3,  /* XOR one byte with 0xFF                        */
    OP_BYTEFLIP_2    =  4,  /* XOR two consecutive bytes with 0xFF           */
    OP_BYTEFLIP_4    =  5,  /* XOR four consecutive bytes with 0xFF          */
    OP_ARITH_INC_8   =  6,  /* Add 1 to a random byte (uint8 wrap)           */
    OP_ARITH_DEC_8   =  7,  /* Subtract 1 from a random byte                 */
    OP_ARITH_INC_16  =  8,  /* Add 1 to a random aligned uint16              */
    OP_ARITH_DEC_16  =  9,  /* Subtract 1 from a random aligned uint16       */
    OP_INTERESTING_8 = 10,  /* Replace a byte with an interesting value       */
    OP_BLOCK_DEL     = 11,  /* Delete a random-length byte block              */
    OP_BLOCK_INS     = 12,  /* Insert a random byte at a random offset        */
    OP_SPLICE        = 13,  /* Append bytes from add_buf (cross-seed splice)  */
    TOTAL_OPERATORS  = 14   /* Sentinel — do not use as an action ID          */
} mutator_op_t;

/* Interesting 8-bit values that often trigger parser boundary conditions.
 * Prefixed MUT_ to avoid collision with AFL++ config.h INTERESTING_8 macro. */
static const int8_t MUT_INTERESTING_8[] = {
    -128, -1, 0, 1, 16, 32, 64, 100, 127
};
#define MUT_INTERESTING_8_LEN \
    ((int)(sizeof(MUT_INTERESTING_8) / sizeof(MUT_INTERESTING_8[0])))

/* -------------------------------------------------------------------------
 * mutate_apply – Apply one mutation operator to a buffer in-place.
 *
 * The operator index (op) selects which transformation to apply.
 * Random offset selection is decoupled from strategy: rand() is called
 * inside this function so the caller does not need to manage offsets.
 *
 * @param op         Operator ID from mutator_op_t.
 * @param buf        Input/output buffer (mutated in-place).
 * @param size       Current buffer size in bytes.
 * @param add_buf    Secondary seed buffer for splice operations (may be NULL).
 * @param add_size   Length of add_buf in bytes.
 * @param max_size   Hard upper bound on output size (e.g., Modbus frame cap).
 *
 * @return  New buffer size after mutation (may differ for block ops).
 *          Returns 0 if buf is NULL or size is 0.
 * ------------------------------------------------------------------------- */
size_t mutate_apply(uint16_t op, uint8_t *buf, size_t size,
                    const uint8_t *add_buf, size_t add_size,
                    size_t max_size);

#endif /* OPERATORS_H */
