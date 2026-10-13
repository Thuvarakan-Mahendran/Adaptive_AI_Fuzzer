/*
 * protocol.h – Binary IPC wire protocol between the C mutator and the
 *              Python PPO scheduler daemon.
 *
 * All structs are __attribute__((packed)) to guarantee zero-copy
 * serialization compatible with Python's struct.unpack / np.frombuffer.
 *
 * Transport: UNIX domain socket, SOCK_SEQPACKET
 * Socket path: DEFAULT_SOCKET_PATH (/tmp/afl_ai_bridge.sock)
 *
 * Message flow:
 *   C → Python : MSG_MUT_REQ    (192-D PPO state + deferred reward, 278 bytes)
 *   Python → C : MSG_MUT_RESP   (batch of operator IDs, 66 bytes)
 *   C → Python : MSG_SEED_QUERY (execution trace for seed prioritization, 2057 bytes)
 *   Python → C : MSG_SEED_VERD  (accept/skip verdict + Q-value, 6 bytes)
 */

#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <stdint.h>

/* -------------------------------------------------------------------------
 * Socket path
 * ------------------------------------------------------------------------- */
#define DEFAULT_SOCKET_PATH "/tmp/afl_ai_bridge.sock"
#define DL_SOCKET_PATH      "/tmp/afl_dl_bridge.sock"
#define TRACE_PATH_MAX      256

/* Maximum batch size for operator scheduling */
#define MUT_BATCH_MAX  32

/* Maximum trace steps in a seed query */
#define TRACE_STEPS_MAX 512

/* -------------------------------------------------------------------------
 * Message type tag (first byte of every message)
 * ------------------------------------------------------------------------- */
typedef enum {
    MSG_MUT_REQ      = 0x01,   /* C → Python: PPO state + deferred reward   */
    MSG_MUT_RESP     = 0x02,   /* Python → C: scheduled operator ID batch   */
    MSG_SEED_QUERY   = 0x03,   /* C → Python: execution trace for DL scorer */
    MSG_SEED_VERD    = 0x04,   /* Python → C: seed prioritization verdict   */
    MSG_TRACE_SUBMIT = 0x05,   /* C → Python: seed_id + trace path          */
    MSG_TRACE_SCORE  = 0x06,   /* Python → C: Q-value + accept verdict      */
    MSG_DL_REWARD    = 0x07    /* C → Python: seed outcome reward feedback  */
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
 * Total size: 1 + 4 + 4 + (512 * 4) = 2057 bytes.
 * ------------------------------------------------------------------------- */
struct seed_query_msg {
    uint8_t           type;                       /* MSG_SEED_QUERY (0x03)   */
    uint32_t          seed_id;                    /* Monotonic queue ID      */
    uint32_t          trace_len;                  /* Valid entries in steps[] */
    struct trace_step steps[TRACE_STEPS_MAX];     /* Execution trace buffer  */
} __attribute__((packed));

/* -------------------------------------------------------------------------
 * MSG_SEED_VERD – Python → C
 *
 * Python responds with an accept/skip decision and an optional Q-value
 * estimate to be logged for analysis.
 * Total size: 1 + 4 + 1 = 6 bytes.
 * ------------------------------------------------------------------------- */
struct seed_verdict_msg {
    uint8_t  type;      /* MSG_SEED_VERD (0x04)                              */
    float    q_value;   /* Estimated long-term utility Q(s,a); for logging   */
    uint8_t  accept;    /* 1 = proceed with fuzzing; 0 = skip/deprioritize   */
} __attribute__((packed));

/* -------------------------------------------------------------------------
 * MSG_TRACE_SUBMIT – C → Python
 *
 * Sent by mutator/trace collector to notify the Python DL daemon of a
 * newly generated variable-length trace stored in /dev/shm.
 * Total size: 1 + 4 + 256 = 261 bytes.
 * ------------------------------------------------------------------------- */
struct trace_submit_msg {
    uint8_t  type;                        /* MSG_TRACE_SUBMIT (0x05)         */
    uint32_t seed_id;                     /* Monotonic queue ID / entry idx  */
    char     trace_path[TRACE_PATH_MAX];  /* Path to trace file in /dev/shm */
} __attribute__((packed));

/* -------------------------------------------------------------------------
 * MSG_TRACE_SCORE – Python → C
 *
 * Python DL daemon responds with an estimated Q-value and accept/skip verdict.
 * Total size: 1 + 4 + 1 = 6 bytes.
 * ------------------------------------------------------------------------- */
struct trace_score_msg {
    uint8_t  type;      /* MSG_TRACE_SCORE (0x06)                            */
    float    q_value;   /* Estimated Q(seed) utility for queue ranking       */
    uint8_t  accept;    /* 1 = proceed with fuzzing; 0 = skip/deprioritize   */
} __attribute__((packed));

/* -------------------------------------------------------------------------
 * MSG_DL_REWARD – C → Python
 *
 * Sent after fuzzing a seed finishes to provide real-time feedback
 * directly into the DL replay buffer.
 * Total size: 1 + 4 + 4 + 1 + 4 = 14 bytes.
 * ------------------------------------------------------------------------- */
struct dl_reward_msg {
    uint8_t  type;           /* MSG_DL_REWARD (0x07)                        */
    uint32_t seed_id;        /* Seed queue index that was fuzzed            */
    uint32_t delta_edges;    /* New coverage bitmap edges discovered        */
    uint8_t  found_crash;    /* 1 if new crash observed, else 0             */
    uint32_t total_execs;    /* Cumulative campaign executions counter      */
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
 * Total size: 1 + 4 + 16 + 256 + 1 = 278 bytes.
 * ------------------------------------------------------------------------- */
struct mut_request_msg {
    uint8_t  type;              /* MSG_MUT_REQ (0x01)                         */
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
 * Total size: 1 + 1 + (32 * 2) = 66 bytes.
 * ------------------------------------------------------------------------- */
struct mut_response_msg {
    uint8_t  type;                    /* MSG_MUT_RESP (0x02)                  */
    uint8_t  count;                   /* Number of valid entries in actions[] */
    uint16_t actions[MUT_BATCH_MAX];  /* Operator IDs (mutator_op_t values)   */
} __attribute__((packed));

#endif /* PROTOCOL_H */
