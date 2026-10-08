/*
 * protocol.h – Binary IPC wire protocol between the C mutator and the
 *              Python PPO scheduler daemon.
 *
 * All structs are __attribute__((packed)) to guarantee zero-copy
 * serialization compatible with Python's struct.unpack / np.frombuffer.
 *
 * Transport: UNIX domain socket, SOCK_SEQPACKET
 * Socket path: DEFAULT_SOCKET_PATH
 *
 * Message flow:
 *   C → Python : MSG_MUT_REQ  (192-D PPO state + deferred reward)
 *   Python → C : MSG_MUT_RESP (batch of operator IDs)
 *   C → Python : MSG_SEED_QUERY (execution trace for seed prioritization)
 *   Python → C : MSG_SEED_VERD  (accept/skip verdict)
 */

#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <stdint.h>

/* -------------------------------------------------------------------------
 * Socket path
 * ------------------------------------------------------------------------- */
#define DEFAULT_SOCKET_PATH "/tmp/afl_ai_bridge.sock"

/* Maximum batch size for operator scheduling */
#define MUT_BATCH_MAX  32

/* Maximum trace steps in a seed query */
#define TRACE_STEPS_MAX 512

/* -------------------------------------------------------------------------
 * Message type tag (first byte of every message)
 * ------------------------------------------------------------------------- */
typedef enum {
    MSG_MUT_REQ    = 0x01,   /* C → Python: PPO state + deferred reward   */
    MSG_MUT_RESP   = 0x02,   /* Python → C: scheduled operator ID batch   */
    MSG_SEED_QUERY = 0x03,   /* C → Python: execution trace for DL scorer */
    MSG_SEED_VERD  = 0x04    /* Python → C: seed prioritization verdict   */
} msg_type_t;

/* -------------------------------------------------------------------------
 * Trace step — one branch/edge sample within an execution trace
 * ------------------------------------------------------------------------- */
struct trace_step {
    uint16_t edge_id;        /* Branch identifier (AFL++ XOR-hash of src/dst) */
    uint8_t  hit_count_bin;  /* Logarithmic hit-count bucket [0..7]           */
    uint8_t  fault_flags;    /* Bit 0: crash, Bit 1: hang, Bit 2: timeout     */
} __attribute__((packed));

/* -------------------------------------------------------------------------
 * MSG_SEED_QUERY – C → Python
 *
 * Sent once per newly enqueued seed so the Python DL scorer can decide
 * whether to prioritize, accept, or skip this entry.
 * ------------------------------------------------------------------------- */
struct seed_query_msg {
    uint8_t           type;                       /* MSG_SEED_QUERY          */
    uint32_t          seed_id;                    /* Monotonic queue ID      */
    uint32_t          trace_len;                  /* Valid entries in steps[] */
    struct trace_step steps[TRACE_STEPS_MAX];     /* Execution trace buffer  */
} __attribute__((packed));

/* -------------------------------------------------------------------------
 * MSG_SEED_VERD – Python → C
 *
 * Python responds with an accept/skip decision and an optional Q-value
 * estimate to be logged for analysis.
 * ------------------------------------------------------------------------- */
struct seed_verdict_msg {
    uint8_t  type;      /* MSG_SEED_VERD                                     */
    float    q_value;   /* Estimated long-term utility Q(s,a); for logging   */
    uint8_t  accept;    /* 1 = proceed with fuzzing; 0 = skip/deprioritize   */
} __attribute__((packed));

/* -------------------------------------------------------------------------
 * MSG_MUT_REQ – C → Python
 *
 * Sent each time the local action batch runs out.  Contains the deferred
 * reward R_{t-1} computed by the aggregator from the previous execution,
 * plus the current 192-D PPO state encoded as:
 *
 *   coverage_vec[16]  = 128 binary bits packed into 16 bytes
 *                       (state[0:128] of the 192-D PPO contract)
 *   history_vec[64]   = 64 float32 values covering:
 *                         [0:16]  operator usage  (state[128:144])
 *                         [16:32] operator success (state[144:160])
 *                         [32:48] last-op one-hot  (state[160:176])
 *                         [48:64] campaign telem.  (state[176:192])
 * ------------------------------------------------------------------------- */
struct mut_request_msg {
    uint8_t  type;              /* MSG_MUT_REQ                                */
    float    deferred_reward;   /* R_{t-1} from aggregator snapshot diff      */
    uint8_t  coverage_vec[16];  /* 128 coverage bits packed as 16 bytes       */
    float    history_vec[64];   /* 64-D mutation history + telemetry          */
    uint8_t  queue_cur_changed; /* 1 if AFL++ switched to a new seed          */
} __attribute__((packed));

/* -------------------------------------------------------------------------
 * MSG_MUT_RESP – Python → C
 *
 * Python PPO agent replies with a batch of k operator IDs to be consumed
 * one at a time by successive afl_custom_fuzz() invocations.
 * ------------------------------------------------------------------------- */
struct mut_response_msg {
    uint8_t  type;                    /* MSG_MUT_RESP                         */
    uint8_t  count;                   /* Number of valid entries in actions[] */
    uint16_t actions[MUT_BATCH_MAX];  /* Operator IDs (mutator_op_t values)   */
} __attribute__((packed));

#endif /* PROTOCOL_H */
