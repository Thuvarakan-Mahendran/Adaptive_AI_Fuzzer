/*
 * mutator.h – AFL++ custom mutator state structure.
 *
 * One mutator_state_t is allocated in afl_custom_init() and freed in
 * afl_custom_deinit().  It persists across the entire fuzzing campaign.
 *
 * Networking is fully decoupled into ipc_client_t from ipc_bridge.
 */

#ifndef MUTATOR_H
#define MUTATOR_H

#include "../../AFLplusplus/include/afl-fuzz.h"
#include "protocol.h"
#include "aggregator.h"
#include "operators.h"
#include "socket_client.h"
#include "trace_collector.h"

/* Maximum frame size passed to the mutator.
 * Set to 256 to match the Modbus RTU frame cap used in this project.
 * Increase if targeting larger protocols. */
#define MUTATOR_MAX_SIZE 256

/* Number of operator IDs requested per IPC round-trip.
 * Larger batches reduce socket overhead; smaller batches react faster to
 * coverage changes.  8 is a sensible default. */
#define MUTATOR_BATCH_SIZE 8

/* -------------------------------------------------------------------------
 * mutator_state_t – Persistent per-campaign mutator state
 * ------------------------------------------------------------------------- */
typedef struct {
    afl_state_t   *afl;         /* Pointer to live AFL++ state (not owned)   */
    aggregator_t   agg;         /* Feedback aggregator (snapshot + history)  */
    ipc_client_t  *ipc;         /* Decoupled IPC client handle for PPO       */
    ipc_client_t  *dl_ipc;      /* Decoupled IPC client handle for DL daemon */
    trace_buffer_t trace_buf;   /* Buffer for storing execution traces       */

    uint8_t       *out_buf;     /* Heap buffer for the mutated output frame  */
    size_t         max_size;    /* Hard cap on output size (MUTATOR_MAX_SIZE)*/

    /* Batch action cache: PPO returns up to MUT_BATCH_MAX IDs at once.
     * They are consumed one per afl_custom_fuzz() invocation. */
    uint16_t       action_queue[MUT_BATCH_MAX];
    uint8_t        action_count;/* Number of valid entries in action_queue   */
    uint8_t        action_idx;  /* Next index to consume                     */
} mutator_state_t;

#endif /* MUTATOR_H */
