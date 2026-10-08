/*
 * aggregator.h – Feedback Aggregator interface.
 *
 * The aggregator decouples raw AFL++ engine telemetry from the PPO state
 * construction.  It maintains a snapshot of the previous execution so it
 * can compute the deferred reward R_{t-1}, project the AFL++ bitmap onto
 * the 128-bit coverage representation, and track the 64-D decaying
 * mutation-history vector.
 *
 * No AFL++ headers are required here; callers fill an afl_snapshot_t from
 * live afl_state_t fields and pass it in.  This keeps the aggregator unit-
 * testable without a running fuzzer.
 */

#ifndef AGGREGATOR_H
#define AGGREGATOR_H

#include "protocol.h"
#include <stdint.h>
#include <stddef.h>

/* AFL++ standard bitmap size in bytes (64 KB = 65 536 bytes = 2^16 bits) */
#define AFL_BITMAP_BYTES 65536

/* Number of history float slots (matches protocol.h history_vec[64]) */
#define HISTORY_DIM 64

/* EMA decay factor for operator usage history (not paper-specified) */
#define HISTORY_DECAY 0.95f

/* -------------------------------------------------------------------------
 * afl_snapshot_t – Decoupled abstraction of one point-in-time AFL++ state.
 *
 * Callers construct this from live afl_state_t / fsrv fields and pass a
 * const pointer to aggregator functions.  The aggregator never accesses
 * afl_state_t directly.
 * ------------------------------------------------------------------------- */
typedef struct {
    const uint8_t *trace_bits;   /* Pointer to AFL++ SHM coverage bitmap     */
    size_t         bitmap_size;  /* Actual bitmap size in bytes (≤65536)     */
    uint64_t       saved_crashes;/* afl->saved_crashes at snapshot time      */
    uint64_t       saved_hangs;  /* afl->saved_hangs at snapshot time        */
    uint32_t       current_entry;/* afl->current_entry (seed queue index)    */
} afl_snapshot_t;

/* -------------------------------------------------------------------------
 * aggregator_t – Persistent aggregator state between executions.
 *
 * One instance lives in mutator_state_t for the lifetime of a campaign.
 * ------------------------------------------------------------------------- */
typedef struct {
    uint8_t  prev_bitmap[AFL_BITMAP_BYTES]; /* Bitmap at last snapshot       */
    uint64_t prev_crashes;                  /* Crash count at last snapshot  */
    uint64_t prev_hangs;                    /* Hang count at last snapshot   */
    uint32_t prev_entry;                    /* Queue index at last snapshot  */
    float    history[HISTORY_DIM];          /* Decaying operator history     */
    uint64_t total_execs;                   /* Monotonic execution counter   */
} aggregator_t;

/* -------------------------------------------------------------------------
 * Public API
 * ------------------------------------------------------------------------- */

/**
 * agg_init – Zero-initialise an aggregator.
 * Must be called once before any other aggregator function.
 */
void agg_init(aggregator_t *agg);

/**
 * agg_fold_bitmap – Project a full AFL++ bitmap onto 16 bytes (128 bits).
 *
 * Each set byte in trace_bits maps its index modulo 128 to a bit in
 * out_cov[16].  Hash collisions are harmless (OR semantics).
 *
 * @param trace_bits  Pointer to raw AFL++ coverage bitmap.
 * @param size        Bitmap length in bytes.
 * @param out_cov     Output: 16-byte array (128 binary bits, packed).
 */
void agg_fold_bitmap(const uint8_t *trace_bits, size_t size,
                     uint8_t out_cov[16]);

/**
 * agg_compute_reward – Compute the deferred reward R_{t-1} and update
 *                      the stored snapshot baseline.
 *
 * Reward formula (matches ppo_scheduler/config.py):
 *   R = norm_cov + LAMBDA_CRASH * I_crash + LAMBDA_NOVELTY * I_novel
 *
 * norm_cov = delta_coverage_bits / log(e + total_execs)
 *
 * After computation the aggregator adopts snap as the new baseline.
 *
 * @param agg   Persistent aggregator state (modified in-place).
 * @param snap  Current AFL++ snapshot.
 * @return      Scalar reward float.
 */
float agg_compute_reward(aggregator_t *agg, const afl_snapshot_t *snap);

/**
 * agg_update_history – Apply one EMA decay step and credit the chosen op.
 *
 * history[i] *= HISTORY_DECAY  for all i
 * history[executed_op] += 1.0   (if executed_op < HISTORY_DIM)
 *
 * @param agg          Aggregator state (modified in-place).
 * @param executed_op  Operator ID that was just applied.
 */
void agg_update_history(aggregator_t *agg, uint16_t executed_op);

/**
 * agg_build_mut_request – Populate a mut_request_msg from current state.
 *
 * Folds the bitmap and copies the history vector; sets queue_cur_changed
 * by comparing snap->current_entry with the previously stored entry.
 *
 * @param agg             Aggregator state (read-only; snapshot NOT updated).
 * @param snap            Current AFL++ snapshot.
 * @param deferred_reward Pre-computed R_{t-1} from agg_compute_reward().
 * @param out_msg         Output message to fill (caller-allocated).
 */
void agg_build_mut_request(aggregator_t *agg, const afl_snapshot_t *snap,
                            float deferred_reward,
                            struct mut_request_msg *out_msg);

#endif /* AGGREGATOR_H */
