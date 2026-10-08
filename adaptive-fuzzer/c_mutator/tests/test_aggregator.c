/*
 * tests/test_aggregator.c – Synthetic out-of-band unit tests for the
 *                            Feedback Aggregator submodule.
 *
 * Compile and run independently (no AFL++ required):
 *   make test_agg && ./test_agg
 *
 * Tests:
 *   T1: Zero bitmap → zero delta coverage → novelty=0 → reward=0
 *   T2: New edges + crash → reward >= LAMBDA_CRASH + LAMBDA_NOVELTY
 *   T3: bitmap fold output is non-zero when active edges present
 *   T4: Consecutive calls use correct baseline (no double-counting)
 *   T5: History vector decays and accumulates correctly
 *   T6: agg_build_mut_request populates all fields correctly
 */

#include "../aggregator.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>

/* ---- Helpers ---------------------------------------------------------- */

#define PASS(name) do { printf("[PASS] %s\n", (name)); } while (0)
#define FAIL(name, msg) do { \
    fprintf(stderr, "[FAIL] %s: %s\n", (name), (msg)); \
    return 1; \
} while (0)

static int floats_close(float a, float b, float eps)
{
    float diff = a - b;
    return (diff < eps && diff > -eps);
}

/* ---- Test 1: Zero bitmap reward --------------------------------------- */

static int test_zero_reward(void)
{
    aggregator_t agg;
    agg_init(&agg);

    uint8_t zero_bitmap[AFL_BITMAP_BYTES];
    memset(zero_bitmap, 0, sizeof(zero_bitmap));

    afl_snapshot_t snap = {
        .trace_bits    = zero_bitmap,
        .bitmap_size   = AFL_BITMAP_BYTES,
        .saved_crashes = 0,
        .saved_hangs   = 0,
        .current_entry = 0,
    };

    float r = agg_compute_reward(&agg, &snap);

    /* No coverage, no crash, no novelty → R should be 0 */
    if (!floats_close(r, 0.0f, 1e-5f))
        FAIL("test_zero_reward", "Expected reward=0 for zero bitmap");

    PASS("test_zero_reward");
    return 0;
}

/* ---- Test 2: Crash + novelty reward ----------------------------------- */

static int test_crash_novelty_reward(void)
{
    aggregator_t agg;
    agg_init(&agg);

    /* Step 0: prime the baseline with the zero bitmap */
    uint8_t bitmap_t0[AFL_BITMAP_BYTES];
    memset(bitmap_t0, 0, sizeof(bitmap_t0));
    afl_snapshot_t snap0 = {
        .trace_bits    = bitmap_t0,
        .bitmap_size   = AFL_BITMAP_BYTES,
        .saved_crashes = 0,
    };
    agg_compute_reward(&agg, &snap0);

    /* Step 1: 5 new edges + crash */
    uint8_t bitmap_t1[AFL_BITMAP_BYTES];
    memset(bitmap_t1, 0, sizeof(bitmap_t1));
    for (int i = 0; i < 5; i++)
        bitmap_t1[i * 100] = 1;   /* 5 non-zero bytes at offsets 0,100,...400 */

    afl_snapshot_t snap1 = {
        .trace_bits    = bitmap_t1,
        .bitmap_size   = AFL_BITMAP_BYTES,
        .saved_crashes = 1,        /* New crash detected */
        .saved_hangs   = 0,
        .current_entry = 0,
    };

    float r1 = agg_compute_reward(&agg, &snap1);

    /* R >= LAMBDA_CRASH(10) + LAMBDA_NOVELTY(1) = 11, plus norm_cov >= 0 */
    if (r1 < 11.0f)
        FAIL("test_crash_novelty_reward",
             "Expected reward >= 11.0 (crash=10 + novelty=1)");

    PASS("test_crash_novelty_reward");
    return 0;
}

/* ---- Test 3: Bitmap fold non-zero ------------------------------------- */

static int test_fold_nonzero(void)
{
    uint8_t bitmap[AFL_BITMAP_BYTES];
    memset(bitmap, 0, sizeof(bitmap));
    bitmap[0]   = 1;    /* Bit 0 → bin 0 → out_cov[0] bit 0 */
    bitmap[100] = 1;    /* Bit 100 → bin 100 → out_cov[12] bit 4 */
    bitmap[127] = 1;    /* Bit 127 → bin 127 → out_cov[15] bit 7 */

    uint8_t folded[16];
    agg_fold_bitmap(bitmap, AFL_BITMAP_BYTES, folded);

    /* At least some folded bits must be set */
    int any_set = 0;
    for (int i = 0; i < 16; i++)
        if (folded[i]) any_set = 1;

    if (!any_set)
        FAIL("test_fold_nonzero", "Expected non-zero folded output");

    /* Byte 0 must have bit 0 set (bitmap[0]=1 → index 0 % 128 = 0) */
    if (!(folded[0] & 0x01))
        FAIL("test_fold_nonzero", "Bit 0 not set in folded[0]");

    PASS("test_fold_nonzero");
    return 0;
}

/* ---- Test 4: No double-counting across consecutive calls -------------- */

static int test_no_double_count(void)
{
    aggregator_t agg;
    agg_init(&agg);

    uint8_t bm0[AFL_BITMAP_BYTES], bm1[AFL_BITMAP_BYTES];
    memset(bm0, 0, sizeof(bm0));
    memset(bm1, 0, sizeof(bm1));
    bm1[42] = 1;   /* One new edge in step 1 */

    afl_snapshot_t s0 = { .trace_bits = bm0, .bitmap_size = AFL_BITMAP_BYTES };
    afl_snapshot_t s1 = { .trace_bits = bm1, .bitmap_size = AFL_BITMAP_BYTES };

    agg_compute_reward(&agg, &s0);
    float r1 = agg_compute_reward(&agg, &s1);  /* 1 new edge */
    float r2 = agg_compute_reward(&agg, &s1);  /* Same bitmap again → 0 new */

    if (r2 >= r1 && r2 > 0.01f)
        FAIL("test_no_double_count",
             "Second call with same bitmap should not produce novelty reward");

    PASS("test_no_double_count");
    return 0;
}

/* ---- Test 5: History decay -------------------------------------------- */

static int test_history_decay(void)
{
    aggregator_t agg;
    agg_init(&agg);

    /* Credit operator 3 once */
    agg_update_history(&agg, 3);
    float after_first = agg.history[3];

    /* Decay once without crediting */
    agg_update_history(&agg, 0);   /* Credits op 0, decays all */
    float after_decay = agg.history[3];

    if (after_first <= 0.0f)
        FAIL("test_history_decay", "History[3] should be > 0 after credit");

    if (after_decay >= after_first)
        FAIL("test_history_decay", "History[3] should decay after next call");

    /* Op 0 should now be > 0 */
    if (agg.history[0] <= 0.0f)
        FAIL("test_history_decay", "History[0] should be > 0 after credit");

    PASS("test_history_decay");
    return 0;
}

/* ---- Test 6: agg_build_mut_request fields ----------------------------- */

static int test_build_mut_request(void)
{
    aggregator_t agg;
    agg_init(&agg);

    /* Prime baseline */
    uint8_t bm0[AFL_BITMAP_BYTES];
    memset(bm0, 0, sizeof(bm0));
    afl_snapshot_t s0 = { .trace_bits = bm0, .bitmap_size = AFL_BITMAP_BYTES,
                           .current_entry = 0 };
    agg_compute_reward(&agg, &s0);

    /* Build request for a new snapshot */
    uint8_t bm1[AFL_BITMAP_BYTES];
    memset(bm1, 0, sizeof(bm1));
    bm1[10] = 1;
    afl_snapshot_t s1 = { .trace_bits = bm1, .bitmap_size = AFL_BITMAP_BYTES,
                           .current_entry = 5 };

    struct mut_request_msg msg;
    memset(&msg, 0, sizeof(msg));
    agg_build_mut_request(&agg, &s1, 1.0f, &msg);

    if (msg.type != MSG_MUT_REQ)
        FAIL("test_build_mut_request", "msg.type != MSG_MUT_REQ");

    if (msg.deferred_reward != 1.0f)
        FAIL("test_build_mut_request", "deferred_reward mismatch");

    /* current_entry changed from 0 → 5, so queue_cur_changed must be 1 */
    if (msg.queue_cur_changed != 1)
        FAIL("test_build_mut_request", "queue_cur_changed should be 1");

    /* coverage_vec must have at least one non-zero byte */
    int cov_set = 0;
    for (int i = 0; i < 16; i++) if (msg.coverage_vec[i]) cov_set = 1;
    if (!cov_set)
        FAIL("test_build_mut_request", "coverage_vec is all zero unexpectedly");

    PASS("test_build_mut_request");
    return 0;
}

/* ---- Main ------------------------------------------------------------- */

int main(void)
{
    int failures = 0;

    failures += test_zero_reward();
    failures += test_crash_novelty_reward();
    failures += test_fold_nonzero();
    failures += test_no_double_count();
    failures += test_history_decay();
    failures += test_build_mut_request();

    if (failures == 0) {
        printf("\n[PASS] Aggregator synthetic unit tests succeeded. (%d/6)\n",
               6 - failures);
        return 0;
    } else {
        fprintf(stderr, "\n[FAIL] %d test(s) failed.\n", failures);
        return 1;
    }
}
