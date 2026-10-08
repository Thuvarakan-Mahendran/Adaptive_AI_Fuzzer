/*
 * aggregator.c – Feedback Aggregator implementation.
 *
 * Implements snapshot diffing, 128-bit bitmap projection, deferred reward
 * computation (R_{t-1}), and the decaying operator-history vector.
 *
 * Design note: bitmap comparisons are done 64 bits at a time using
 * __builtin_popcountll for speed.  The full 65 536-byte bitmap is covered
 * in 8 192 iterations of the inner loop.
 */

#include "aggregator.h"

#include <string.h>
#include <math.h>

#ifndef M_E
#  define M_E 2.71828182845904523536
#endif

/* Reward coefficients — must match ppo_scheduler/config.py */
#define LAMBDA_CRASH   10.0f
#define LAMBDA_NOVELTY  1.0f

/* -------------------------------------------------------------------------
 * agg_init
 * ------------------------------------------------------------------------- */
void agg_init(aggregator_t *agg)
{
    memset(agg, 0, sizeof(aggregator_t));
}

/* -------------------------------------------------------------------------
 * agg_fold_bitmap
 *
 * Maps each set byte in trace_bits to a bit position in out_cov[16]:
 *   bit_idx  = index_of_byte % 128
 *   out_cov[bit_idx / 8] |= (1 << (bit_idx % 8))
 *
 * This is a simple modulo projection; hash collisions (multiple edges
 * mapping to the same bin) are intentional and expected.
 * ------------------------------------------------------------------------- */
void agg_fold_bitmap(const uint8_t *trace_bits, size_t size,
                     uint8_t out_cov[16])
{
    memset(out_cov, 0, 16);
    if (!trace_bits || size == 0)
        return;

    for (size_t i = 0; i < size; i++) {
        if (trace_bits[i]) {
            uint8_t bit_idx = (uint8_t)(i % 128);
            out_cov[bit_idx >> 3] |= (uint8_t)(1u << (bit_idx & 7u));
        }
    }
}

/* -------------------------------------------------------------------------
 * agg_compute_reward
 *
 * Reward formula:
 *   delta_cov  = popcount( cur_bitmap & ~prev_bitmap )   [new bits set]
 *   norm_cov   = delta_cov / log(e + total_execs)        [saturation guard]
 *   I_crash    = 1 if snap->saved_crashes > agg->prev_crashes
 *   I_novel    = 1 if delta_cov > 0
 *   R          = norm_cov + LAMBDA_CRASH*I_crash + LAMBDA_NOVELTY*I_novel
 *
 * After computing the reward, the aggregator adopts snap as the new baseline
 * (prev_bitmap is updated, prev_crashes/hangs/entry are stored).
 * ------------------------------------------------------------------------- */
float agg_compute_reward(aggregator_t *agg, const afl_snapshot_t *snap)
{
    uint32_t delta_cov = 0;

    if (snap && snap->trace_bits) {
        /* Compare current and previous bitmap 8 bytes at a time */
        const uint64_t *cur64  = (const uint64_t *)snap->trace_bits;
        const uint64_t *prev64 = (const uint64_t *)agg->prev_bitmap;
        size_t words = (snap->bitmap_size < AFL_BITMAP_BYTES)
                       ? snap->bitmap_size / 8
                       : AFL_BITMAP_BYTES / 8;

        for (size_t i = 0; i < words; i++) {
            uint64_t novel = cur64[i] & (~prev64[i]);
            if (novel)
                delta_cov += (uint32_t)__builtin_popcountll(novel);
        }
    }

    agg->total_execs++;

    /* Log-normalize to guard against reward growth as coverage saturates */
    float norm_cov = (float)delta_cov
                     / logf((float)(M_E + (double)agg->total_execs));

    float i_crash  = (snap && snap->saved_crashes > agg->prev_crashes)
                     ? 1.0f : 0.0f;
    float i_novel  = (delta_cov > 0) ? 1.0f : 0.0f;

    float reward = norm_cov
                   + LAMBDA_CRASH   * i_crash
                   + LAMBDA_NOVELTY * i_novel;

    /* Update baseline snapshot */
    if (snap && snap->trace_bits) {
        size_t copy_bytes = (snap->bitmap_size < AFL_BITMAP_BYTES)
                            ? snap->bitmap_size : AFL_BITMAP_BYTES;
        memcpy(agg->prev_bitmap, snap->trace_bits, copy_bytes);
    }
    if (snap) {
        agg->prev_crashes = snap->saved_crashes;
        agg->prev_hangs   = snap->saved_hangs;
        agg->prev_entry   = snap->current_entry;
    }

    return reward;
}

/* -------------------------------------------------------------------------
 * agg_update_history
 *
 * EMA decay on all history slots, then credit the executed operator.
 * history[i] *= HISTORY_DECAY   ∀ i
 * history[executed_op] += 1.0   if executed_op < HISTORY_DIM
 * ------------------------------------------------------------------------- */
void agg_update_history(aggregator_t *agg, uint16_t executed_op)
{
    for (int i = 0; i < HISTORY_DIM; i++)
        agg->history[i] *= HISTORY_DECAY;

    if (executed_op < HISTORY_DIM)
        agg->history[executed_op] += 1.0f;
}

/* -------------------------------------------------------------------------
 * agg_build_mut_request
 *
 * Assembles a struct mut_request_msg ready to be sent over the socket.
 * Does NOT update any aggregator state (snapshot update happens in
 * agg_compute_reward; history update happens in agg_update_history).
 * ------------------------------------------------------------------------- */
void agg_build_mut_request(aggregator_t *agg, const afl_snapshot_t *snap,
                            float deferred_reward,
                            struct mut_request_msg *out_msg)
{
    out_msg->type             = MSG_MUT_REQ;
    out_msg->deferred_reward  = deferred_reward;
    out_msg->queue_cur_changed = (snap && snap->current_entry != agg->prev_entry)
                                 ? 1u : 0u;

    /* Pack 128-bit folded coverage into coverage_vec[16] */
    if (snap && snap->trace_bits)
        agg_fold_bitmap(snap->trace_bits, snap->bitmap_size,
                        out_msg->coverage_vec);
    else
        memset(out_msg->coverage_vec, 0, sizeof(out_msg->coverage_vec));

    /* Copy 64-D history vector verbatim into history_vec */
    memcpy(out_msg->history_vec, agg->history,
           HISTORY_DIM * sizeof(float));
}
