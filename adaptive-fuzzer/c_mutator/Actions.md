# c_mutator — Actions Reference

## Purpose

`c_mutator/` implements the AFL++ custom mutator shared library (`libmutator.so`) that bridges the C-based fuzzing loop with the Python PPO scheduling daemon.  It exposes four AFL++ hooks and routes operator-selection decisions through a UNIX domain socket IPC channel.

---

## Files & Functions

### `protocol.h`
Wire-protocol definition shared between C mutator and Python daemon.

| Symbol | Kind | Purpose |
|---|---|---|
| `DEFAULT_SOCKET_PATH` | macro | Socket path `/tmp/afl_ai_bridge.sock` |
| `MUT_BATCH_MAX` | macro | Maximum actions per IPC response (32) |
| `TRACE_STEPS_MAX` | macro | Max trace steps in a seed query (512) |
| `msg_type_t` | enum | Message type tags (0x01–0x04) |
| `trace_step` | struct (packed) | One branch/edge sample (4 bytes) |
| `seed_query_msg` | struct (packed) | MSG_SEED_QUERY payload (2057 bytes) |
| `seed_verdict_msg` | struct (packed) | MSG_SEED_VERD payload (6 bytes) |
| `mut_request_msg` | struct (packed) | MSG_MUT_REQ payload (278 bytes) |
| `mut_response_msg` | struct (packed) | MSG_MUT_RESP payload (66 bytes) |

---

### `aggregator.h` / `aggregator.c`
Feedback aggregator: snapshot diffing, reward computation, history vector management.

| Function | Purpose |
|---|---|
| `agg_init(agg, trace_bits, map_size)` | Initialize aggregator, snapshot zero bitmap |
| `agg_snapshot(agg)` | Save current bitmap state (deferred reward baseline) |
| `agg_compute_reward(agg)` | Compute R_{t-1} = 10·Δcrashes + 1·Δnovelty_bits since last snapshot |
| `agg_update_history(agg, op_id, success)` | Decay history vector and add usage/success for `op_id` |
| `agg_build_mut_request(agg, msg, queue_changed)` | Fill a `mut_request_msg` ready to send over IPC |

**Design:** Coverage bitmap is compared 64 bits at a time using `__builtin_popcountll` over the 65 536-byte AFL++ bitmap (8 192 loop iterations).

---

### `operators.h` / `operators.c`
Frozen C mutation operator pool.

| Symbol | Kind | Purpose |
|---|---|---|
| `mutator_op_t` | enum | 14 operator IDs (0–13); must remain stable across builds |
| `MUT_INTERESTING_8[]` | const array | 9 boundary-triggering byte values |
| `mutate_apply(op, buf, size, add_buf, add_size, max_size)` | function | Apply one operator to buffer in-place; returns new size |

**Operators (action space):**

| ID | Name | Description |
|---|---|---|
| 0 | `OP_BITFLIP_1_1` | Flip 1 bit |
| 1 | `OP_BITFLIP_2_1` | Flip 2 consecutive bits |
| 2 | `OP_BITFLIP_4_1` | Flip 4 consecutive bits (nybble) |
| 3 | `OP_BYTEFLIP_1` | XOR one byte with 0xFF |
| 4 | `OP_BYTEFLIP_2` | XOR two consecutive bytes with 0xFF |
| 5 | `OP_BYTEFLIP_4` | XOR four consecutive bytes with 0xFF |
| 6 | `OP_ARITH_INC_8` | +1 to a random byte |
| 7 | `OP_ARITH_DEC_8` | -1 from a random byte |
| 8 | `OP_ARITH_INC_16` | +1 to a random aligned uint16 |
| 9 | `OP_ARITH_DEC_16` | -1 from a random aligned uint16 |
| 10 | `OP_INTERESTING_8` | Replace byte with interesting value |
| 11 | `OP_BLOCK_DEL` | Delete a random-length block |
| 12 | `OP_BLOCK_INS` | Insert a random byte at random offset |
| 13 | `OP_SPLICE` | Append bytes from secondary seed |

---

### `mutator.h` / `mutator.c`
AFL++ custom mutator entry points and IPC socket management.

| Function | AFL++ Hook | Purpose |
|---|---|---|
| `afl_custom_init(afl, seed)` | required | Allocate `mutator_state_t`, connect socket, init aggregator |
| `afl_custom_fuzz(data, buf, buf_size, out_buf, add_buf, add_size, max_size)` | required | Pop operator from batch cache (or fetch new batch via IPC), apply mutation |
| `afl_custom_queue_get(data, filename)` | optional | Send `MSG_SEED_QUERY` for new seed; return accept verdict |
| `afl_custom_deinit(data)` | required | Close socket, free output buffer and state struct |

**Internal helpers:**

| Helper | Purpose |
|---|---|
| `try_connect_socket()` | Create + connect SEQPACKET socket; sets 100 ms SO_RCVTIMEO |
| `request_batch(st)` | Build + send MSG_MUT_REQ; receive + parse MSG_MUT_RESP into action_queue |

---

### `tests/test_aggregator.c`
Unit test binary for the aggregator (compiled as `test_agg`).

| Test | What it verifies |
|---|---|
| `test_zero_reward` | Reward is 0.0 when bitmap and crash/hang counters don't change |
| `test_crash_novelty_reward` | Reward correctly adds crash (×10) + novelty (×1) contributions |
| `test_fold_nonzero` | `agg_snapshot` folds new bits into seen_bits; re-snap gives zero novelty |
| `test_no_double_count` | Second snap without changes gives 0 reward (no double-counting) |
| `test_history_decay` | History vector decays toward zero after N update cycles |
| `test_build_mut_request` | `agg_build_mut_request` sets correct msg type and coverage bytes |

---

## Process Flow

```
AFL++ fuzzing loop
       │
       ▼
afl_custom_init()
  ├─ malloc(mutator_state_t)
  ├─ try_connect_socket() → /tmp/afl_ai_bridge.sock
  ├─ agg_init()   (snapshot clean bitmap)
  └─ malloc(out_buf, MUTATOR_MAX_SIZE=256)

       │ per-seed
       ▼
afl_custom_queue_get()
  ├─ send MSG_SEED_QUERY  (seed_id + execution trace)
  ├─ recv MSG_SEED_VERD   (accept=1 → fuzz; accept=0 → skip)
  └─ return accept verdict to AFL++

       │ per-mutation
       ▼
afl_custom_fuzz()
  ├─ if action_queue empty:
  │    agg_snapshot()         (save bitmap baseline)
  │    request_batch(st)      (IPC round-trip)
  │      ├─ agg_build_mut_request() → send MSG_MUT_REQ
  │      └─ recv MSG_MUT_RESP → fill action_queue[8]
  │    [fallback: if IPC fails → random local operator]
  ├─ pop action_queue[action_idx++]
  ├─ memcpy input → out_buf
  ├─ mutate_apply(op, out_buf, size, add_buf, add_size, max_size)
  └─ return new size to AFL++

       │ campaign end
       ▼
afl_custom_deinit()
  ├─ close(sock_fd)
  └─ free(out_buf), free(mutator_state_t)
```

---

## Commands (in order)

### 1. Build

```bash
cd /home/captainalan2/Research/adaptive-fuzzer/c_mutator
make -B all
```

Expected output:
```
clang-14 -O3 -fPIC -Wall -Wextra -I../../AFLplusplus/include -c aggregator.c -o aggregator.o
clang-14 -O3 -fPIC -Wall -Wextra -I../../AFLplusplus/include -c operators.c -o operators.o
clang-14 -O3 -fPIC -Wall -Wextra -I../../AFLplusplus/include -c mutator.c -o mutator.o
clang-14 -shared -lm -o libmutator.so aggregator.o operators.o mutator.o
[c_mutator] Built libmutator.so
clang-14 -O3 -fPIC ... tests/test_aggregator.c aggregator.o -o test_agg -lm
[c_mutator] Built test_agg
```

### 2. Run unit tests

```bash
./test_agg
```

Expected:
```
[PASS] test_zero_reward
[PASS] test_crash_novelty_reward
[PASS] test_fold_nonzero
[PASS] test_no_double_count
[PASS] test_history_decay
[PASS] test_build_mut_request

[PASS] Aggregator synthetic unit tests succeeded. (6/6)
```

### 3. Start mock IPC daemon (background)

```bash
python3 /home/captainalan2/Research/adaptive-fuzzer/scripts/run_mock_daemon.py -v &
```

### 4. Load environment and run AFL++

```bash
source /home/captainalan2/Research/adaptive-fuzzer/scripts/env.sh
afl-fuzz -i corpus/base64 -o out/base64 -m none -- \
    build/native/fuzzer_base64 @@
```

### 5. Clean

```bash
make -C /home/captainalan2/Research/adaptive-fuzzer/c_mutator clean
```

---

## Dependencies

| Dependency | Version | Path |
|---|---|---|
| clang-14 | LLVM 14 | `/usr/bin/clang-14` |
| AFL++ headers | pinned | `/home/captainalan2/Research/AFLplusplus/include/` |
| Python 3 | ≥ 3.8 | system |
| libm | system | linked via `-lm` |

> [!IMPORTANT]
> `libmutator.so` includes `afl-fuzz.h` (via `mutator.h`) only for the `afl_state_t` type definition.
> The AFL++ source tree at `../../AFLplusplus/` must be present when building `c_mutator/`.
> AFL++ itself does **not** need to be compiled first — only the headers are required.

---

## IPC Integration Test Plan

> This section describes the integration test procedure for connecting `libmutator.so` to the mock daemon end-to-end.

### Components needed

| Component | Role |
|---|---|
| `libmutator.so` | AFL++ custom mutator |
| `scripts/run_mock_daemon.py` | Python mock PPO daemon |
| `build/native/fuzzer_base64` or `fuzzer_modbus_native` | Instrumented fuzz target |
| `corpus/base64/` or equivalent seed corpus | AFL++ initial seeds |
| `afl-fuzz` | `/usr/local/bin/afl-fuzz` |

### Test procedure

```bash
# Step 1: Build the mutator
cd /home/captainalan2/Research/adaptive-fuzzer/c_mutator && make -B all

# Step 2: Start the mock daemon in the background
python3 /home/captainalan2/Research/adaptive-fuzzer/scripts/run_mock_daemon.py -v \
    > /tmp/mock_daemon.log 2>&1 &
DAEMON_PID=$!

# Step 3: Load environment
source /home/captainalan2/Research/adaptive-fuzzer/scripts/env.sh

# Step 4: Run AFL++ for ~30 seconds (Ctrl-C to stop)
timeout 30 afl-fuzz \
    -i /home/captainalan2/Research/adaptive-fuzzer/corpus/base64 \
    -o /tmp/afl_out_ipc_test \
    -m none \
    -- /home/captainalan2/Research/adaptive-fuzzer/build/native/fuzzer_base64 @@ \
    2>&1 | tee /tmp/afl_ipc_test.log || true

# Step 5: Verify IPC messages were exchanged
echo "=== Daemon log ==="
cat /tmp/mock_daemon.log | grep -E "MSG_MUT_REQ|MSG_MUT_RESP|connected"
echo "=== AFL log ==="
grep -i "mutator\|custom" /tmp/afl_ipc_test.log || echo "(no custom mutator lines in AFL log)"

# Step 6: Stop daemon
kill $DAEMON_PID 2>/dev/null
```

### Success criteria

1. Daemon log shows `Client connected` within 2 seconds of `afl-fuzz` start.
2. At least one `MSG_MUT_REQ` / `MSG_MUT_RESP` exchange appears in daemon log.
3. AFL++ does not crash or stall; `afl-fuzz` runs for the full 30 seconds.
4. `libmutator.so` stderr (captured in AFL log) shows `[mutator] connected to daemon`.
5. If daemon is killed mid-run, AFL++ continues fuzzing (fallback policy).

### Known failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `[mutator] socket connect failed` on startup | Daemon not started yet | Start daemon before AFL++ |
| AFL++ exits with `Illegal instruction` | `afl-clang-fast` used for `.so` | Rebuild with `clang-14` |
| `recv() error: [Errno 11]` in daemon log | EAGAIN on RCVTIMEO | Normal; mutator falls back locally |
| Zero IPC exchanges in daemon log | `AFL_CUSTOM_MUTATOR_LIBRARY` not set | Source `env.sh` first |

---

## Test Results

| Test | Status | Notes |
|---|---|---|
| `test_zero_reward` | ✅ PASS | Aggregator unit test |
| `test_crash_novelty_reward` | ✅ PASS | Aggregator unit test |
| `test_fold_nonzero` | ✅ PASS | Aggregator unit test |
| `test_no_double_count` | ✅ PASS | Aggregator unit test |
| `test_history_decay` | ✅ PASS | Aggregator unit test |
| `test_build_mut_request` | ✅ PASS | Aggregator unit test |
| `libmutator.so` link | ✅ Built (21K) | Built using `clang-14 -fPIC -shared` |
| Mock daemon startup | ✅ PASS | Binds `/tmp/afl_ai_bridge.sock`, handles SIGTERM cleanly |
| Python struct sizes | ✅ Match C sizeof() | 278 / 66 / 6 / 2057 bytes verified against C ABI |
| IPC Round-Trip Test | ✅ PASS | Verified `MSG_MUT_REQ` → `MSG_MUT_RESP` and `MSG_SEED_QUERY` → `MSG_SEED_VERD` |
| AFL++ Live Integration | ✅ PASS | AFL++ loaded `libmutator.so`, hooked fuzz loops, exchanged 13 batches via IPC with daemon, 10 new corpus entries |

---

## Future Integration Work

1. **Connect real PPO daemon:** Replace `run_mock_daemon.py` with the trained PPO scheduler (`ppo_scheduler/`). The daemon must accept SEQPACKET connections and respond with `MSG_MUT_RESP` using the same wire format.

2. **`ppo_scheduler` side:** The `ppo_scheduler/run_daemon.py` (to be written) should:
   - Load a trained model checkpoint from `ppo_scheduler/checkpoints/`
   - Listen on `DEFAULT_SOCKET_PATH`
   - Unpack `mut_request_msg` into a 192-D numpy state vector
   - Call `PPOAgent.select_action()` and return the action batch

3. **State normalization:** The aggregator's `history_vec` uses raw counts and floats. The PPO model expects normalized inputs (zero-mean, unit-variance). Add a normalization layer in the daemon, not in the C mutator.

4. **Multi-instance fuzzing (`-M/-S` mode):** Currently one daemon serves one AFL++ instance. For parallel fuzzing, either:
   - Run one daemon per instance (different socket paths), or
   - Make the daemon multi-threaded (one thread per accepted connection).

5. **Persistent state across restarts:** Add checkpoint save/load to the daemon so PPO training state survives daemon restarts.

6. **Corpus seed prioritization:** The `afl_custom_queue_get()` hook currently sends full trace data. Wire up the PPO's DL scorer to actually read `trace_steps[]` and compute a priority score.
