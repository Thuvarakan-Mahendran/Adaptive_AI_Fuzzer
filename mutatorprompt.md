# Implementation Guide: AFL++ Custom Mutator with Feedback Aggregator

This document outlines the step-by-step technical implementation to build, unit-test, and validate the external C mutator (`libmutator.so`) embedding the Feedback Aggregator. It links AFL++ (`~/Research/AFLplusplus`) with the AI learning modules in `~/Research/adaptive-fuzzer` via high-performance UNIX domain sockets (`SOCK_SEQPACKET`).

The already existing files may have an approach, the below given folder, file structure can be adapted accordingly. For example there is this Makefile under c_mutator but there may be a Makefile which is inside adaptive-fuzzer/ folder. Always analyze and act accordingly. Check before make any changes. don't blindly make changes or create anything.

---

## 1. Directory Structure Setup

Organize the implementation within `~/Research/adaptive-fuzzer/` to cleanly separate the C native extension, wire protocols, and testing harnesses:

```bash
~/Research/
├── AFLplusplus/                  # Pinned AFL++ source tree (v4.08c)
└── adaptive-fuzzer/
    ├── c_mutator/                # Native C/C++ Mutator & Aggregator
    │   ├── Makefile
    │   ├── protocol.h            # Binary IPC wire protocol definitions
    │   ├── aggregator.h          # Feedback Aggregator interfaces & types
    │   ├── aggregator.c          # Snapshot diffing, bitmap projection, reward logic
    │   ├── operators.h           # Mutator operator declarations
    │   ├── operators.c           # C mutation pool implementation
    │   ├── mutator.h             # Mutator state & API structures
    │   ├── mutator.c             # AFL++ Custom Mutator entry points
    │   └── tests/
    │       └── test_aggregator.c # Synthetic unit testing harness (out-of-band)
    ├── scripts/
    │   ├── env.sh                # Campaign environment variables
    │   └── run_mock_daemon.py    # Python IPC validation & stub server
    └── DECISIONS.md              # Engineering decision log
```

## 2. Task Decomposition Overview

### Task 1: Wire Protocol & IPC Contract (`protocol.h`)

- **Subtask 1.1:** Struct layout for PPO mutation requests and responses (`MSG_MUT_REQ`, `MSG_MUT_RESP`).
- **Subtask 1.2:** Struct layout for DL seed prioritization queries and verdicts (`MSG_SEED_QUERY`, `MSG_SEED_VERD`).
- **Subtask 1.3:** Enforce strict byte alignment (`__attribute__((packed))`).

### Task 2: Feedback Aggregator Submodule (`aggregator.h`, `aggregator.c`)

- **Subtask 2.1:** Snapshot definition and decoupled abstraction (`afl_snapshot_t`).
- **Subtask 2.2:** 128-bit hash-folded bitmap projection.
- **Subtask 2.3:** 64-dimensional decaying mutation history vector.
- **Subtask 2.4:** Deferred credit assignment (\(R_{t-1}\)) and coverage normalization.
- **Subtask 2.5:** Out-of-band synthetic unit tests (`test_aggregator.c`).

### Task 3: C Mutation Operator Pool (`operators.h`, `operators.c`)

- **Subtask 3.1:** Enumerate and freeze integer action space.
- **Subtask 3.2:** Implement bit/byte flips, arithmetics, interesting values, block ops, and splice.
- **Subtask 3.3:** Decouple operator transformation from uniform offset sampling.
- **Subtask 3.4:** Manage heap buffers and enforce `max_size` bounds (e.g., 256 bytes).

### Task 4: Custom Mutator Engine & IPC Integration (`mutator.h`, `mutator.c`)

- **Subtask 4.1:** `afl_custom_init` with socket setup, `FD_CLOEXEC`, `SO_RCVTIMEO`, and retry loop.
- **Subtask 4.2:** `afl_custom_queue_get` hooking for DL seed queries (`MSG_SEED_QUERY`).
- **Subtask 4.3:** `afl_custom_fuzz` hooking for PPO scheduling (`MSG_MUT_REQ`) and batching (\(k\)).
- **Subtask 4.4:** Resilience fallbacks on socket timeouts or disconnects.
- **Subtask 4.5:** Clean deallocation in `afl_custom_deinit`.

### Task 5: Compilation, Environment Scripting, and Mock Verification

- **Subtask 5.1:** Build `libmutator.so` against `AFLplusplus/include`.
- **Subtask 5.2:** Configure `scripts/env.sh` with `AFL_CUSTOM_MUTATOR_ONLY=1`.
- **Subtask 5.3:** End-to-end integration test with Python stub listener (`scripts/run_mock_daemon.py`).

---

## 3. Detailed Step-by-Step Implementation

Task given here is just as an example, create the relevant code according to the analysis you had, given code can involve changes accordingly. No strict follow up, change according to the need.

## Task 1: Wire Protocol & IPC Contract (`protocol.h`)

Create `c_mutator/protocol.h`. Pack all structures to guarantee zero-copy serialization between C and Python (`np.frombuffer` / `struct.unpack`).

```c
#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <stdint.h>

#define DEFAULT_SOCKET_PATH "/tmp/afl_ai_bridge.sock"

typedef enum {
    MSG_MUT_REQ    = 0x01,  // C -> Python: PPO State & Deferred Reward
    MSG_MUT_RESP   = 0x02,  // Python -> C: Scheduled Operator Indices
    MSG_SEED_QUERY = 0x03,  // C -> Python: Seed Trace Query for DL Optimizer
    MSG_SEED_VERD  = 0x04   // Python -> C: DL Seed Verdict
} msg_type_t;

// Transition step within the execution trace for DL Optimizer
struct trace_step {
    uint16_t edge_id;        // Branch identifier (XOR hash)
    uint8_t  hit_count_bin;  // Logarithmic bucket
    uint8_t  fault_flags;    // Invariant violations / trap indicators
} __attribute__((packed));

// Seed evaluation request
struct seed_query_msg {
    uint8_t           type;          // MSG_SEED_QUERY
    uint32_t          seed_id;       // Monotonic queue entry ID
    uint32_t          trace_len;     // Number of trace steps
    struct trace_step steps[512];    // Fixed trace buffer
} __attribute__((packed));

// DL Optimizer verdict response
struct seed_verdict_msg {
    uint8_t  type;                   // MSG_SEED_VERD
    float    q_value;                // Predicted long-term utility Q(s, a)
    uint8_t  accept;                 // 1 = fuzz; 0 = skip/deprioritize
} __attribute__((packed));

// PPO Mutation scheduling request
struct mut_request_msg {
    uint8_t  type;                   // MSG_MUT_REQ
    float    deferred_reward;        // R_{t-1} computed via snapshot diff
    uint8_t  coverage_vec[16];       // 128-bit hash-folded bitmap
    float    history_vec[64];        // 64-dim operator decay vector
    uint8_t  queue_cur_changed;      // 1 if AFL++ switched seeds
} __attribute__((packed));

// PPO Mutation batch response
struct mut_response_msg {
    uint8_t  type;                   // MSG_MUT_RESP
    uint8_t  count;                  // Batch size k (e.g., 8-32)
    uint16_t actions[32];            // Array of selected operator IDs
} __attribute__((packed));

#endif // PROTOCOL_H
```

## Task 2: Feedback Aggregator Submodule (`aggregator.h`, `aggregator.c`)

### `c_mutator/aggregator.h`

Abstract raw engine telemetry to decouple state construction from live AFL++ state:

```c
#ifndef AGGREGATOR_H
#define AGGREGATOR_H

#include "protocol.h"
#include <stdint.h>
#include <stddef.h>

typedef struct {
    const uint8_t *trace_bits;
    size_t         bitmap_size;
    uint64_t       saved_crashes;
    uint64_t       saved_hangs;
    uint32_t       current_entry;
} afl_snapshot_t;

typedef struct {
    uint8_t  prev_bitmap[65536];
    uint64_t prev_crashes;
    uint64_t prev_hangs;
    uint32_t prev_entry;
    float    history[64];
    uint64_t total_execs;
} aggregator_t;

void  agg_init(aggregator_t *agg);
void  agg_fold_bitmap(const uint8_t *trace_bits, size_t size, uint8_t out_cov[16]);
float agg_compute_reward(aggregator_t *agg, const afl_snapshot_t *snap);
void  agg_update_history(aggregator_t *agg, uint16_t executed_op);
void  agg_build_mut_request(aggregator_t *agg, const afl_snapshot_t *snap,
                            float deferred_reward, struct mut_request_msg *out_msg);

#endif // AGGREGATOR_H
```

### `c_mutator/aggregator.c`

Implement reward calculation (Equation 1), log-normalization, and vector folding:

```c
#include "aggregator.h"
#include <string.h>
#include <math.h>

#ifndef M_E
#define M_E 2.71828182845904523536
#endif

void agg_init(aggregator_t *agg) {
    memset(agg, 0, sizeof(aggregator_t));
}

// 128-bit hash-folding projection
void agg_fold_bitmap(const uint8_t *trace_bits, size_t size, uint8_t out_cov[16]) {
    memset(out_cov, 0, 16);
    if (!trace_bits) return;
    for (size_t i = 0; i < size; i++) {
        if (trace_bits[i]) {
            uint16_t bit_idx = (uint16_t)(i % 128);
            out_cov[bit_idx / 8] |= (uint8_t)(1 << (bit_idx % 8));
        }
    }
}

// Deferred reward calculation (Equation 1)
float agg_compute_reward(aggregator_t *agg, const afl_snapshot_t *snap) {
    uint32_t delta_cov = 0;
    const uint64_t *cur64 = (const uint64_t *)snap->trace_bits;
    const uint64_t *prev64 = (const uint64_t *)agg->prev_bitmap;

    if (snap->trace_bits) {
        for (size_t i = 0; i < 65536 / 8; i++) {
            uint64_t novel = cur64[i] & (~prev64[i]);
            if (novel) {
                delta_cov += (uint32_t)__builtin_popcountll(novel);
            }
        }
    }

    agg->total_execs++;
    float norm_cov = (float)delta_cov / logf((float)(M_E + agg->total_execs));

    float lambda_crash = 10.0f;
    float i_crash = (snap->saved_crashes > agg->prev_crashes) ? 1.0f : 0.0f;
    float lambda_novelty = 1.0f;
    float i_novel = (delta_cov > 0) ? 1.0f : 0.0f;

    float r_deferred = norm_cov + (lambda_crash * i_crash) + (lambda_novelty * i_novel);

    // Update snapshot baseline
    if (snap->trace_bits) {
        memcpy(agg->prev_bitmap, snap->trace_bits, 65536);
    }
    agg->prev_crashes = snap->saved_crashes;
    agg->prev_hangs   = snap->saved_hangs;
    agg->prev_entry   = snap->current_entry;

    return r_deferred;
}

void agg_update_history(aggregator_t *agg, uint16_t executed_op) {
    const float decay = 0.95f;
    for (int i = 0; i < 64; i++) {
        agg->history[i] *= decay;
    }
    if (executed_op < 64) {
        agg->history[executed_op] += 1.0f;
    }
}

void agg_build_mut_request(aggregator_t *agg, const afl_snapshot_t *snap,
                            float deferred_reward, struct mut_request_msg *out_msg) {
    out_msg->type = MSG_MUT_REQ;
    out_msg->deferred_reward = deferred_reward;
    out_msg->queue_cur_changed = (snap->current_entry != agg->prev_entry) ? 1 : 0;
    agg_fold_bitmap(snap->trace_bits, snap->bitmap_size, out_msg->coverage_vec);
    memcpy(out_msg->history_vec, agg->history, sizeof(out_msg->history_vec));
}
```

### Synthetic Unit Test (`c_mutator/tests/test_aggregator.c`)

Compile independently to catch off-by-one labeling and bitmap reduction errors before running fuzzing campaigns:

```c
#include "../aggregator.h"
#include <stdio.h>
#include <assert.h>

int main(void) {
    aggregator_t agg;
    agg_init(&agg);

    uint8_t mock_bitmap_t0[65536] = {0};
    uint8_t mock_bitmap_t1[65536] = {0};

    // Step t: mark 5 new edges
    for (int i = 0; i < 5; i++) mock_bitmap_t1[i * 100] = 1;

    afl_snapshot_t snap0 = { .trace_bits = mock_bitmap_t0, .bitmap_size = 65536, .saved_crashes = 0 };
    afl_snapshot_t snap1 = { .trace_bits = mock_bitmap_t1, .bitmap_size = 65536, .saved_crashes = 1 };

    agg_compute_reward(&agg, &snap0);
    float r1 = agg_compute_reward(&agg, &snap1);

    // Verify crash indicator and novelty inclusion
    assert(r1 >= 11.0f); // 10.0 (crash) + 1.0 (novelty) + norm_cov

    uint8_t folded[16];
    agg_fold_bitmap(mock_bitmap_t1, 65536, folded);
    assert(folded[0] != 0);

    printf("[PASS] Aggregator synthetic unit tests succeeded.\n");
    return 0;
}
```

## Task 3: C Mutation Operator Pool (`operators.h`, `operators.c`)

### `c_mutator/operators.h`

Freeze action indices for PPO:

```c
#ifndef OPERATORS_H
#define OPERATORS_H

#include <stdint.h>
#include <stddef.h>

typedef enum {
    OP_BITFLIP_1_1   = 0,
    OP_BITFLIP_2_1   = 1,
    OP_BITFLIP_4_1   = 2,
    OP_BYTEFLIP_1    = 3,
    OP_BYTEFLIP_2    = 4,
    OP_BYTEFLIP_4    = 5,
    OP_ARITH_INC_8   = 6,
    OP_ARITH_DEC_8   = 7,
    OP_ARITH_INC_16  = 8,
    OP_ARITH_DEC_16  = 9,
    OP_INTERESTING_8 = 10,
    OP_BLOCK_DEL     = 11,
    OP_BLOCK_INS     = 12,
    OP_SPLICE        = 13,
    TOTAL_OPERATORS  = 14
} mutator_op_t;

size_t mutate_apply(uint16_t op, uint8_t *buf, size_t size,
                    const uint8_t *add_buf, size_t add_size, size_t max_size);

#endif // OPERATORS_H
```

### `c_mutator/operators.c`

Implement transformations while decoupling strategy from random offset selection:

```c
#include "operators.h"
#include <stdlib.h>
#include <string.h>

static const int8_t interesting_8[] = { -128, -1, 0, 1, 16, 32, 64, 100, 127 };

size_t mutate_apply(uint16_t op, uint8_t *buf, size_t size,
                    const uint8_t *add_buf, size_t add_size, size_t max_size) {
    if (!buf || size == 0) return 0;
    size_t off = (size_t)rand() % size;

    switch (op) {
        case OP_BITFLIP_1_1:
            buf[off] ^= (uint8_t)(1 << (rand() % 8));
            return size;
        case OP_BYTEFLIP_1:
            buf[off] ^= 0xFF;
            return size;
        case OP_ARITH_INC_8:
            buf[off] += 1;
            return size;
        case OP_ARITH_DEC_8:
            buf[off] -= 1;
            return size;
        case OP_INTERESTING_8:
            buf[off] = (uint8_t)interesting_8[rand() % (sizeof(interesting_8) / sizeof(interesting_8[0]))];
            return size;
        case OP_BLOCK_DEL:
            if (size > 2) {
                size_t del_len = 1 + ((size_t)rand() % (size - off));
                memmove(buf + off, buf + off + del_len, size - (off + del_len));
                return size - del_len;
            }
            return size;
        case OP_BLOCK_INS:
            if (size < max_size) {
                memmove(buf + off + 1, buf + off, size - off);
                buf[off] = (uint8_t)rand();
                return size + 1;
            }
            return size;
        case OP_SPLICE:
            if (add_buf && add_size > 0 && size < max_size) {
                size_t copy_len = (size + add_size <= max_size) ? add_size : (max_size - size);
                memcpy(buf + size, add_buf, copy_len);
                return size + copy_len;
            }
            buf[off] ^= 0x55;
            return size;
        default:
            buf[off] ^= 0x55;
            return size;
    }
}
```

## Task 4: Custom Mutator Engine & IPC Integration (`mutator.h`, `mutator.c`)

### `c_mutator/mutator.h`

Maintain state across AFL++ hook invocations:

```c
#ifndef MUTATOR_H
#define MUTATOR_H

#include "../AFLplusplus/include/afl-fuzz.h"
#include "protocol.h"
#include "aggregator.h"
#include "operators.h"

typedef struct {
    afl_state_t  *afl;
    aggregator_t  agg;
    int           sock_fd;
    uint8_t      *out_buf;
    size_t        max_size;

    // Batch action cache
    uint16_t      action_queue[32];
    uint8_t       action_count;
    uint8_t       action_idx;
} mutator_state_t;

#endif // MUTATOR_H
```

### `c_mutator/mutator.c`

Bind lifecycle hooks with fallback handling:

```c
#include "mutator.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

mutator_state_t *afl_custom_init(afl_state_t *afl, unsigned int seed) {
    srand(seed);
    mutator_state_t *state = calloc(1, sizeof(mutator_state_t));
    if (!state) return NULL;

    state->afl = afl;
    state->max_size = 256; // Modbus frame cap
    state->out_buf = malloc(state->max_size);
    agg_init(&state->agg);

    // Socket Initialization
    state->sock_fd = socket(AF_UNIX, SOCK_SEQPACKET, 0);
    if (state->sock_fd >= 0) {
        fcntl(state->sock_fd, F_SETFD, FD_CLOEXEC);

        struct timeval tv = { .tv_sec = 0, .tv_usec = 100000 }; // 100ms timeout
        setsockopt(state->sock_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, DEFAULT_SOCKET_PATH, sizeof(addr.sun_path) - 1);

        if (connect(state->sock_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
            close(state->sock_fd);
            state->sock_fd = -1; // Fallback to local mode
        }
    }
    return state;
}

uint8_t afl_custom_queue_get(mutator_state_t *data, const uint8_t *filename) {
    if (!data || data->sock_fd < 0) return 1;

    struct seed_query_msg query = {
        .type = MSG_SEED_QUERY,
        .seed_id = data->afl->current_entry,
        .trace_len = 1,
        .steps[0] = { .edge_id = 0x1, .hit_count_bin = 1, .fault_flags = 0 }
    };

    if (send(data->sock_fd, &query, sizeof(query), MSG_NOSIGNAL) <= 0) return 1;

    struct seed_verdict_msg verd;
    ssize_t ret = recv(data->sock_fd, &verd, sizeof(verd), 0);
    if (ret == sizeof(verd) && verd.type == MSG_SEED_VERD) {
        return verd.accept; // 1 = process; 0 = skip
    }
    return 1;
}

size_t afl_custom_fuzz(mutator_state_t *data, uint8_t *buf, size_t buf_size,
                       uint8_t **out_buf, uint8_t *add_buf, size_t add_buf_size,
                       size_t max_size) {
    if (!data || !buf || buf_size == 0) return 0;

    size_t cur_len = (buf_size > data->max_size) ? data->max_size : buf_size;
    memcpy(data->out_buf, buf, cur_len);

    // Query RL Scheduler if the action batch is depleted
    if (data->action_idx >= data->action_count) {
        afl_snapshot_t snap = {
            .trace_bits    = data->afl->fsrv.trace_bits,
            .bitmap_size   = data->afl->fsrv.map_size ? data->afl->fsrv.map_size : 65536,
            .saved_crashes = data->afl->saved_crashes,
            .saved_hangs   = data->afl->saved_hangs,
            .current_entry = data->afl->current_entry
        };

        float r_deferred = agg_compute_reward(&data->agg, &snap);
        struct mut_request_msg req;
        agg_build_mut_request(&data->agg, &snap, r_deferred, &req);

        struct mut_response_msg resp;
        if (data->sock_fd >= 0 &&
            send(data->sock_fd, &req, sizeof(req), MSG_NOSIGNAL) > 0 &&
            recv(data->sock_fd, &resp, sizeof(resp), 0) == sizeof(resp) &&
            resp.type == MSG_MUT_RESP && resp.count > 0) {
            data->action_count = (resp.count > 32) ? 32 : resp.count;
            memcpy(data->action_queue, resp.actions, data->action_count * sizeof(uint16_t));
        } else {
            // Socket error or timeout fallback
            data->action_count = 1;
            data->action_queue[0] = (uint16_t)(rand() % TOTAL_OPERATORS);
        }
        data->action_idx = 0;
    }

    uint16_t chosen_op = data->action_queue[data->action_idx++];
    agg_update_history(&data->agg, chosen_op);

    size_t mutated_len = mutate_apply(chosen_op, data->out_buf, cur_len,
                                      add_buf, add_buf_size, data->max_size);

    *out_buf = data->out_buf;
    return mutated_len;
}

void afl_custom_deinit(mutator_state_t *data) {
    if (!data) return;
    if (data->sock_fd >= 0) close(data->sock_fd);
    free(data->out_buf);
    free(data);
}
```

## Task 5: Compilation, Environment Scripting, and Mock Verification

### `c_mutator/Makefile`

Target Clang 14 and link against local AFL++ headers:

```makefile
CC = clang
CFLAGS = -O3 -fPIC -Wall -Wextra -I../../AFLplusplus/include
LDFLAGS = -shared -lm

all: libmutator.so test_agg

libmutator.so: aggregator.o operators.o mutator.o
	$(CC) $(LDFLAGS) -o $@ $^

aggregator.o: aggregator.c aggregator.h protocol.h
	$(CC) $(CFLAGS) -c $< -o $@

operators.o: operators.c operators.h
	$(CC) $(CFLAGS) -c $< -o $@

mutator.o: mutator.c mutator.h protocol.h aggregator.h operators.h
	$(CC) $(CFLAGS) -c $< -o $@

test_agg: tests/test_aggregator.c aggregator.o
	$(CC) $(CFLAGS) $^ -o $@ -lm

clean:
	rm -f *.o *.so test_agg
```

### `scripts/env.sh`

Export campaign variables:

```bash
#!/usr/bin/env bash
export AFL_CUSTOM_MUTATOR_LIBRARY="/home/captainalan2/Research/adaptive-fuzzer/c_mutator/libmutator.so"
export AFL_CUSTOM_MUTATOR_ONLY=1    # Disables native havoc to prevent state skew
export AFL_NO_UI=1
export AFL_QUIET=1
```

### Mock Socket Python Listener (`scripts/run_mock_daemon.py`)

Run out-of-band to verify datagram serialization and round-trip communication:

```python
#!/usr/bin/env python3
import socket
import os
import struct

SOCK_PATH = "/tmp/afl_ai_bridge.sock"
if os.path.exists(SOCK_PATH):
    os.remove(SOCK_PATH)

server = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
server.bind(SOCK_PATH)
server.listen(1)
print(f"[*] Mock bridge listening on {SOCK_PATH}...")

conn, _ = server.accept()
print("[+] Fuzzer connected.")

while True:
    data = conn.recv(4096)
    if not data:
        break
    msg_type = data[0]

    if msg_type == 0x01:  # MSG_MUT_REQ
        # Replying with a batch of k=8 random operations
        resp = struct.pack("!BB", 0x02, 8) + struct.pack("!8H", 0, 1, 3, 6, 7, 10, 11, 12)
        # Pad to remaining 32 uint16 slot bounds if required by packing
        resp += b'\x00' * (24 * 2)
        conn.send(resp)

    elif msg_type == 0x03:  # MSG_SEED_QUERY
        # Accept seed: verdict=1, q_value=0.85
        resp = struct.pack("!BfB", 0x04, 0.85, 1)
        conn.send(resp)

conn.close()
server.close()
```

---

## 4. Verification & Exit Criteria

### Synthetic Unit Test

```bash
cd ~/Research/adaptive-fuzzer/c_mutator && make test_agg && ./test_agg
```

**Expected:** Prints `[PASS] Aggregator synthetic unit tests succeeded.`

### IPC Integration Test

According to the existing project setup, provide plan on integration test, needed components and the way it has to be tested

## 5. Create Actions.md file

create a markdown file, providing what has been developed such as functions, the purpose of those functions and the objective it fulfills, process flow and the commands used in the right order to create this module, dependancies and tools installed and the way the module was tested and the future works to be done to integrate.
