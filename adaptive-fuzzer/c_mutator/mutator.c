/*
 * mutator.c – AFL++ Custom Mutator entry points with decoupled IPC integration.
 *
 * Implements the four AFL++ custom mutator hooks:
 *   afl_custom_init()       – allocate state, initialize IPC client
 *   afl_custom_queue_get()  – seed prioritization via MSG_SEED_QUERY
 *   afl_custom_fuzz()       – PPO-scheduled mutation via MSG_MUT_REQ
 *   afl_custom_deinit()     – cleanup
 *
 * All POSIX socket logic has been completely decoupled into the
 * socket_client API (from ipc_bridge).
 *
 * Resilience policy:
 *   If the socket client is disconnected or experiences a timeout/error,
 *   the mutator immediately falls back to a random operator from the local pool.
 */

#include "mutator.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* -------------------------------------------------------------------------
 * Internal helpers
 * ------------------------------------------------------------------------- */

/**
 * fallback_action – Pick one random action when the daemon is unavailable.
 */
static uint16_t fallback_action(void)
{
    return (uint16_t)((unsigned)rand() % TOTAL_OPERATORS);
}

/**
 * replenish_action_queue – Send a MUT_REQ via ipc_client and fill action_queue
 *                          from MUT_RESP, or fall back to a random action.
 */
static void replenish_action_queue(mutator_state_t *state)
{
    /* Build the current AFL++ snapshot */
    afl_snapshot_t snap = {
        .trace_bits    = state->afl->fsrv.trace_bits,
        .bitmap_size   = state->afl->fsrv.map_size
                         ? (size_t)state->afl->fsrv.map_size
                         : AFL_BITMAP_BYTES,
        .saved_crashes = state->afl->saved_crashes,
        .saved_hangs   = state->afl->saved_hangs,
        .current_entry = state->afl->current_entry,
    };

    /* Compute deferred reward from previous execution */
    float r_deferred = agg_compute_reward(&state->agg, &snap);

    /* Assemble the request message */
    struct mut_request_msg req;
    memset(&req, 0, sizeof(req));
    agg_build_mut_request(&state->agg, &snap, r_deferred, &req);

    /* Attempt IPC round-trip via decoupled socket client */
    struct mut_response_msg resp;
    memset(&resp, 0, sizeof(resp));

    if (ipc_client_send_mut_req(state->ipc, &req) &&
        ipc_client_recv_mut_resp(state->ipc, &resp)) {

        if (resp.count > 0) {
            uint8_t cnt = (resp.count > MUT_BATCH_MAX) ? MUT_BATCH_MAX : resp.count;
            state->action_count = cnt;
            memcpy(state->action_queue, resp.actions, (size_t)cnt * sizeof(uint16_t));
            state->action_idx   = 0;
            return;
        }
    }

    /* Fallback: single random operator */
    state->action_count    = 1;
    state->action_queue[0] = fallback_action();
    state->action_idx      = 0;
}

/* -------------------------------------------------------------------------
 * AFL++ Custom Mutator hooks
 * ------------------------------------------------------------------------- */

/**
 * afl_custom_init – Allocate and initialise the mutator.
 *
 * Called once when AFL++ loads libmutator.so.
 */
mutator_state_t *afl_custom_init(afl_state_t *afl, unsigned int seed)
{
    srand(seed);

    mutator_state_t *state = (mutator_state_t *)calloc(1, sizeof(mutator_state_t));
    if (!state) {
        perror("[mutator] calloc failed");
        return NULL;
    }

    state->afl      = afl;
    state->max_size = MUTATOR_MAX_SIZE;
    state->out_buf  = (uint8_t *)malloc(state->max_size);
    if (!state->out_buf) {
        perror("[mutator] malloc out_buf failed");
        free(state);
        return NULL;
    }

    agg_init(&state->agg);

    /* Initialize decoupled IPC client (100ms timeout, 5 retries) */
    state->ipc = ipc_client_init(DEFAULT_SOCKET_PATH, 100, 5);
    if (!ipc_client_is_connected(state->ipc)) {
        fprintf(stderr, "[mutator] daemon not found — running in fallback mode\n");
    } else {
        fprintf(stderr, "[mutator] connected to daemon at %s\n", DEFAULT_SOCKET_PATH);
    }

    /* Initialise action cache as empty so the first fuzz call triggers replenishment */
    state->action_count = 0;
    state->action_idx   = 0;

    return state;
}

/**
 * afl_custom_queue_get – Seed prioritization hook.
 *
 * Called by AFL++ when it is about to switch to a new queue entry.
 * Sends a MSG_SEED_QUERY via ipc_client and returns accept/skip verdict.
 */
uint8_t afl_custom_queue_get(mutator_state_t *data, const uint8_t *filename)
{
    (void)filename;   /* Not used; seed is identified by queue index */

    if (!data || !ipc_client_is_connected(data->ipc))
        return 1;   /* Accept by default in fallback mode */

    struct seed_query_msg query;
    memset(&query, 0, sizeof(query));
    query.type      = MSG_SEED_QUERY;
    query.seed_id   = data->afl->current_entry;
    query.trace_len = 1;
    query.steps[0].edge_id       = 0x0001;
    query.steps[0].hit_count_bin = 1;
    query.steps[0].fault_flags   = 0;

    struct seed_verdict_msg verd;
    memset(&verd, 0, sizeof(verd));

    if (ipc_client_send_seed_query(data->ipc, &query) &&
        ipc_client_recv_seed_verd(data->ipc, &verd)) {
        return verd.accept;
    }

    return 1;   /* Accept on malformed/timeout response */
}

/**
 * afl_custom_fuzz – Main mutation hook.
 *
 * Called by AFL++ once per fuzz iteration.  Consumes one operator ID from
 * the action batch cache (replenishing via IPC when the cache is empty),
 * then applies the selected operator to the input buffer.
 */
size_t afl_custom_fuzz(mutator_state_t *data, uint8_t *buf, size_t buf_size,
                       uint8_t **out_buf, uint8_t *add_buf,
                       size_t add_buf_size, size_t max_size)
{
    if (!data || !buf || buf_size == 0)
        return 0;

    /* Clamp input to our hard cap */
    size_t cur_len = (buf_size > data->max_size) ? data->max_size : buf_size;
    memcpy(data->out_buf, buf, cur_len);

    /* Replenish batch if exhausted */
    if (data->action_idx >= data->action_count)
        replenish_action_queue(data);

    /* Consume next scheduled operator */
    uint16_t chosen_op = data->action_queue[data->action_idx++];

    /* Update decay history for this operator */
    agg_update_history(&data->agg, chosen_op);

    /* Apply the selected mutation */
    size_t effective_max = (max_size < data->max_size) ? max_size : data->max_size;
    size_t mutated_len   = mutate_apply(chosen_op,
                                        data->out_buf, cur_len,
                                        add_buf, add_buf_size,
                                        effective_max);

    *out_buf = data->out_buf;
    return mutated_len;
}

/**
 * afl_custom_deinit – Release all resources.
 *
 * Called by AFL++ when the fuzzing session ends.
 */
void afl_custom_deinit(mutator_state_t *data)
{
    if (!data)
        return;
    if (data->ipc)
        ipc_client_close(data->ipc);
    free(data->out_buf);
    free(data);
}
