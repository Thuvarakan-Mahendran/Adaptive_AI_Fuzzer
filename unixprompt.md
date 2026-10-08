# prompt.md

# Task: Implementation of Dedicated UNIX Socket IPC Subsystem & Component Refactoring

## 1. Objective and Scope
You are tasked with building a dedicated, robust UNIX domain socket subsystem (`AF_UNIX` with `SOCK_SEQPACKET`) to handle low-latency bidirectional IPC between the native AFL++ process (`libmutator.so`) and the Python learning daemon (`adaptive-fuzzer`). 

You must first inspect the current repository structure and existing implementations (`c_mutator/`, `aggregator.*`, `mutator.*`, `protocol.h`, and any Python RL/PPO scheduler scripts), then cleanly decouple all networking boilerplate from the mutator and scheduler into an isolated, unit-testable IPC component.

---

## 2. Preliminary Step: Codebase Audit & Impact Analysis

Before writing new code, analyze existing files across `~/Research/ppo_scheduler/`, `~/Research/adaptive-fuzzer/` , AFL plus don't have direct communication between python environment decide on that one :
1. **Audit `c_mutator/`**:
   - Inspect `protocol.h`, `mutator.h`, and `mutator.c` for embedded socket handles (`sock_fd`), connection routines, and inline `send`/`recv` calls.
   - Verify that `aggregator.h`/`aggregator.c` remains completely network-agnostic (focusing strictly on telemetry snapshotting, 128-bit bitmap projection, 64-dim history decay, and deferred reward $R_{t-1}$).
2. **Audit Python Modules**:
   - Locate the PPO scheduler and mock server implementations.
   - Identify existing packet decoding (`struct.unpack`, `numpy.frombuffer`) and response dispatch logic to migrate them into a dedicated server module.
3. **Record Refactoring Targets**:
   - Note all files requiring modifications, compiler flags, and include paths.

---

## 3. Directory Layout to Establish

Organize the IPC component in a dedicated directory while refactoring the surrounding modules:

```text
adaptive-fuzzer/
├── ipc_bridge/
│   ├── Makefile                   # Independent build harness for IPC tests
│   ├── common/
│   │   └── protocol.h             # Canonical packed binary structs & message types
│   ├── client_c/
│   │   ├── socket_client.h        # C IPC client interface
│   │   └── socket_client.c        # Client socket lifecycle, timeouts, send/recv
│   ├── server_python/
│   │   ├── __init__.py
│   │   ├── socket_server.py       # Reusable Python AF_UNIX SEQPACKET server
│   │   └── mock_server.py         # Out-of-band mock server for benchmarking
│   └── tests/
│       ├── test_ipc_client.c      # Standalone loopback test binary
│       └── run_ipc_check.sh       # Automated round-trip verification script
├── c_mutator/
│   ├── mutator.h                  # Refactored: replaces raw socket fd with ipc_client_t
│   ├── mutator.c                  # Refactored: uses ipc_client API
│   ├── Makefile                   # Updated: links against client_c/socket_client.o
│   ├── aggregator.h / .c          # Unmodified: remains pure telemetry logic
│   └── operators.h / .c           # Unmodified: remains pure mutation logic
└── ppo_scheduler/ (or python daemon)
    └── ...                        # Refactored: imports ipc_bridge.server_python
```
## Decomposed Implementation Tasks

### Task 1: Centralized Binary Protocol (`ipc_bridge/common/protocol.h`)

- [ ] Move and standardize the packed binary protocol with explicit 1-byte message discriminator tags:

  - `MSG_MUT_REQ (0x01)`:
    ```c
    {
        uint8_t type;
        float deferred_reward;
        uint8_t coverage_vec[16];
        float history_vec[64];
        uint8_t queue_cur_changed;
    }
    ```

  - `MSG_MUT_RESP (0x02)`:
    ```c
    {
        uint8_t type;
        uint8_t count;
        uint16_t actions[32];
    }
    ```

  - `MSG_SEED_QUERY (0x03)`:
    ```c
    {
        uint8_t type;
        uint32_t seed_id;
        uint32_t trace_len;
        struct trace_step steps[512];
    }
    ```

  - `MSG_SEED_VERD (0x04)`:
    ```c
    {
        uint8_t type;
        float q_value;
        uint8_t accept;
    }
    ```

- [ ] Enforce `__attribute__((packed))` on all structs to prevent platform padding variations.
- [ ] Ensure `c_mutator/` includes this single canonical header.

### Task 2: C Client Socket Abstraction (`ipc_bridge/client_c/`)

#### Data Structure & API Design (`socket_client.h`)

- [ ] Define an opaque `ipc_client_t` structure.
- [ ] Expose lifecycle functions:

```c
ipc_client_t *ipc_client_init(const char *socket_path, uint32_t timeout_ms, uint32_t max_retries);
int           ipc_client_send_mut_req(ipc_client_t *client, const struct mut_request_msg *req);
int           ipc_client_recv_mut_resp(ipc_client_t *client, struct mut_response_msg *resp);
int           ipc_client_send_seed_query(ipc_client_t *client, const struct seed_query_msg *query);
int           ipc_client_recv_seed_verd(ipc_client_t *client, struct seed_verdict_msg *verd);
void          ipc_client_close(ipc_client_t *client);
```

#### Implementation & Hardening (`socket_client.c`)

- [ ] Use `socket(AF_UNIX, SOCK_SEQPACKET, 0)`.
- [ ] Set `FD_CLOEXEC` to prevent child forkserver instances from leaking the file descriptor.
- [ ] Apply `SO_RCVTIMEO` using the configured timeout (e.g., 50–100 ms).
- [ ] Use `MSG_NOSIGNAL` on all `send()` invocations to prevent `SIGPIPE` termination when the server disconnects.
- [ ] Implement retry-with-backoff inside `ipc_client_init` to accommodate launch race conditions between AFL++ and Python.
- [ ] Return clear error codes (`0` on failure/timeout, `1` on success) to trigger safe fallback behavior in callers.

### Task 3: Python Server Module (`ipc_bridge/server_python/`)

#### Connection & Socket Management (`socket_server.py`)

- [ ] Implement an `IPCServer` class using `socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)`.
- [ ] Clean up stale socket paths (`unlink`/`remove`) on startup and graceful teardown (`SIGINT`/`SIGTERM`).
- [ ] Provide connection listening, non-blocking `select()` polling, and client handling.

#### Zero-Copy Serialization & Decoding

- [ ] Parse `MSG_MUT_REQ` and `MSG_SEED_QUERY` using `struct.unpack` or `numpy.frombuffer`.
- [ ] Provide pack helper functions returning binary payloads for `MSG_MUT_RESP` and `MSG_SEED_VERD`.

#### Mock Engine for Protocol Testing (`mock_server.py`)

- [ ] Implement a standalone CLI runner responding to requests with uniform random actions and acceptance verdicts to exercise the socket independently.

### Task 4: Refactor C Mutator & Aggregator (`c_mutator/`)

#### Clean Up Mutator State (`mutator.h`)

- [ ] Remove raw socket members (`int sock_fd`, `#include <sys/socket.h>`, `#include <sys/un.h>`).
- [ ] Replace them with a reference to `ipc_client_t *ipc`.

#### Simplify Hook Implementation (`mutator.c`)

- [ ] In `afl_custom_init`: Replace socket initialization blocks with `ipc_client_init(SOCKET_PATH, 100, 5)`.
- [ ] In `afl_custom_queue_get`: Replace inline send/recv with `ipc_client_send_seed_query` and `ipc_client_recv_seed_verd`. Return `1` (accept) on failure.
- [ ] In `afl_custom_fuzz`: Replace inline mutation querying with `ipc_client_send_mut_req` and `ipc_client_recv_mut_resp`. On timeout or error, fall back to local random operator selection.
- [ ] In `afl_custom_deinit`: Clean up via `ipc_client_close`.

#### Update `c_mutator/Makefile`

- [ ] Include `-I../ipc_bridge/common -I../ipc_bridge/client_c`.
- [ ] Compile and link `ipc_bridge/client_c/socket_client.o` directly into `libmutator.so`.

### Task 5: Refactor Python PPO Scheduler Integration

- [ ] Replace custom socket handling in the PPO daemon with `ipc_bridge.server_python.socket_server`.
- [ ] Decouple incoming message handling:

  - Route message tag `0x01` (`MSG_MUT_REQ`) to the PPO transition buffer and action inference pipeline.
  - Route message tag `0x03` (`MSG_SEED_QUERY`) to the DL Seed Optimizer (or mock stub until active).

- [ ] Verify that model training steps running every `$N$` iterations do not block the socket event loop beyond the C-side `SO_RCVTIMEO` threshold (utilize batching or asynchronous worker delegation).

### Task 6: Independent Verification & Benchmarking

#### Standalone Loopback Test (`test_ipc_client.c` + `mock_server.py`)

- [ ] Stream 100,000 synthetic `mut_request_msg` and `seed_query_msg` packets across the bridge.
- [ ] Validate that zero byte corruption occurs and datagram boundaries are preserved.

#### Latency Profiling

- [ ] Measure round-trip time (RTT) distribution.
- [ ] Confirm mean RTT is well below execution limits.

#### Fault Resilience Test

- [ ] Terminate the Python server process while `test_ipc_client` is actively transmitting.
- [ ] Confirm the C client catches timeouts cleanly, executes fallback actions, and exits without segmentation faults or `SIGPIPE` aborts.

#### End-to-End Sanity Run

- [ ] Run `afl-fuzz` using `libmutator.so` against a sample target with `mock_server.py` active to verify smooth campaign execution.

## Exit Criteria

- [ ] `ipc_bridge` builds and passes all tests independently via `make -C ipc_bridge tests`.
- [ ] `libmutator.so` builds cleanly with no raw POSIX socket calls in `mutator.c`.
- [ ] PPO Python server imports `socket_server.py` and processes structured messages without schema mismatches.
- [ ] The system survives abrupt Python server disconnections without crashing AFL++.

## Create Actions.md file

create a markdown file, providing what has been developed such as functions, the purpose of those functions and the objective it fulfills, process flow and the commands used in the right order to create this module, dependancies and tools installed and the way the module was tested, clearly state the testing files where it would be easy to remove later when the full system is ready and the future works to be done to integrate.
