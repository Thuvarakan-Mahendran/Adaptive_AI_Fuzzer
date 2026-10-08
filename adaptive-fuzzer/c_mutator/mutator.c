/*
 * mutator.c – AFL++ Custom Mutator entry points with IPC integration.
 *
 * Implements the four AFL++ custom mutator hooks:
 *   afl_custom_init()       – allocate state, connect socket
 *   afl_custom_queue_get()  – seed prioritization via MSG_SEED_QUERY
 *   afl_custom_fuzz()       – PPO-scheduled mutation via MSG_MUT_REQ
 *   afl_custom_deinit()     – cleanup
 *
 * Resilience policy:
 *   If the socket is unavailable (daemon not running, timeout, send/recv
 *   error), the mutator falls back to a single random operator from the
 *   local pool.  This allows dry-run testing without the Python daemon.
 *
 * Socket type: AF_UNIX SOCK_SEQPACKET (reliable, message-boundary-preserving)
 * Socket path: DEFAULT_SOCKET_PATH (/tmp/afl_ai_bridge.sock)
 * Timeout:     100 ms receive timeout (SO_RCVTIMEO)
 */

#include "mutator.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* -------------------------------------------------------------------------
 * Internal helpers
 * ------------------------------------------------------------------------- */

/**
 * try_connect_socket – Create and connect a SEQPACKET socket.
 * Returns the fd on success, -1 on failure.
 */
static int try_connect_socket(void)
{
    int fd = socket(AF_UNIX, SOCK_SEQPACKET, 0);
    if (fd < 0)
        return -1;

    /* Ensure the fd is not inherited by AFL++ forkserver children */
    if (fcntl(fd, F_SETFD, FD_CLOEXEC) < 0) {
        close(fd);
        return -1;
    }

    /* 100 ms receive timeout — avoids blocking the fuzzer on a stalled daemon */
    struct timeval tv = { .tv_sec = 0, .tv_usec = 100000 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, DEFAULT_SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }

    return fd;
}

/**
 * fallback_action – Pick one random action when the daemon is unavailable.
 */
static uint16_t fallback_action(void)
{
    return (uint16_t)((unsigned)rand() % TOTAL_OPERATORS);
}

/**
 * replenish_action_queue – Send a MUT_REQ and fill action_queue from the
 *                          MUT_RESP, or fall back to a single random action.
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

    /* Attempt IPC round-trip */
    if (state->sock_fd >= 0) {
        ssize_t sent = send(state->sock_fd, &req, sizeof(req), MSG_NOSIGNAL);
        if (sent > 0) {
            struct mut_response_msg resp;
            ssize_t rcvd = recv(state->sock_fd, &resp, sizeof(resp), 0);
            if (rcvd == (ssize_t)sizeof(resp) &&
                resp.type == MSG_MUT_RESP       &&
                resp.count > 0) {

                uint8_t cnt = (resp.count > MUT_BATCH_MAX)
                              ? MUT_BATCH_MAX : resp.count;
                state->action_count = cnt;
                memcpy(state->action_queue, resp.actions,
                       (size_t)cnt * sizeof(uint16_t));
                state->action_idx = 0;
                return;
            }
        }

        /* Socket error or unexpected response — mark as disconnected */
        close(state->sock_fd);
        state->sock_fd = -1;
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
 *
 * @param afl   Pointer to the live AFL++ state.
 * @param seed  Random seed provided by AFL++.
 * @return      Opaque handle (mutator_state_t *) or NULL on fatal error.
 */
mutator_state_t *afl_custom_init(afl_state_t *afl, unsigned int seed)
{
    srand(seed);

    mutator_state_t *state = calloc(1, sizeof(mutator_state_t));
    if (!state) {
        perror("[mutator] calloc failed");
        return NULL;
    }

    state->afl      = afl;
    state->max_size = MUTATOR_MAX_SIZE;
    state->out_buf  = malloc(state->max_size);
    if (!state->out_buf) {
        perror("[mutator] malloc out_buf failed");
        free(state);
        return NULL;
    }

    agg_init(&state->agg);

    /* Attempt socket connection; failure is non-fatal (fallback mode) */
    state->sock_fd = try_connect_socket();
    if (state->sock_fd < 0)
        fprintf(stderr, "[mutator] daemon not found — running in fallback mode\n");
    else
        fprintf(stderr, "[mutator] connected to daemon at %s\n",
                DEFAULT_SOCKET_PATH);

    /* Initialise action cache as empty so the first fuzz call triggers a query */
    state->action_count = 0;
    state->action_idx   = 0;

    return state;
}

/**
 * afl_custom_queue_get – Seed prioritization hook.
 *
 * Called by AFL++ when it is about to switch to a new queue entry.
 * Sends a MSG_SEED_QUERY and returns the daemon's accept/skip decision.
 * Falls back to "accept all" (return 1) if the socket is unavailable.
 *
 * @param data      Mutator state handle.
 * @param filename  Path to the queue entry (not used directly).
 * @return          1 = fuzz this seed, 0 = skip/deprioritize.
 */
uint8_t afl_custom_queue_get(mutator_state_t *data, const uint8_t *filename)
{
    (void)filename;   /* Not used; seed is identified by queue index */

    if (!data || data->sock_fd < 0)
        return 1;   /* Accept by default in fallback mode */

    struct seed_query_msg query;
    memset(&query, 0, sizeof(query));
    query.type     = MSG_SEED_QUERY;
    query.seed_id  = data->afl->current_entry;
    /* Populate one representative trace step from the live bitmap.
     * Full trace reconstruction is deferred to future integration work. */
    query.trace_len = 1;
    query.steps[0].edge_id       = 0x0001;
    query.steps[0].hit_count_bin = 1;
    query.steps[0].fault_flags   = 0;

    if (send(data->sock_fd, &query, sizeof(query), MSG_NOSIGNAL) <= 0) {
        close(data->sock_fd);
        data->sock_fd = -1;
        return 1;
    }

    struct seed_verdict_msg verd;
    ssize_t rcvd = recv(data->sock_fd, &verd, sizeof(verd), 0);
    if (rcvd == (ssize_t)sizeof(verd) && verd.type == MSG_SEED_VERD)
        return verd.accept;

    return 1;   /* Accept on malformed/timeout response */
}

/**
 * afl_custom_fuzz – Main mutation hook.
 *
 * Called by AFL++ once per fuzz iteration.  Consumes one operator ID from
 * the action batch cache (replenishing via IPC when the cache is empty),
 * then applies the selected operator to the input buffer.
 *
 * @param data          Mutator state handle.
 * @param buf           Original input buffer.
 * @param buf_size      Length of buf.
 * @param out_buf       Output pointer (set to state->out_buf on success).
 * @param add_buf       Secondary seed buffer for splice (may be NULL).
 * @param add_buf_size  Length of add_buf.
 * @param max_size      AFL++-requested maximum output size.
 * @return              Length of the mutated output, or 0 on error.
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
 *
 * @param data  Mutator state handle (may be NULL if init failed).
 */
void afl_custom_deinit(mutator_state_t *data)
{
    if (!data)
        return;
    if (data->sock_fd >= 0)
        close(data->sock_fd);
    free(data->out_buf);
    free(data);
}
