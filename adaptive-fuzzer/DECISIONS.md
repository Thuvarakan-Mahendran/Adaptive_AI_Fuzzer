# DECISIONS.md – Engineering Decision Log for `c_mutator`

> Records the rationale for non-obvious technical choices made during the
> implementation of the AFL++ custom mutator (`libmutator.so`) and associated
> IPC infrastructure.

---

## D-01 Compiler: `clang-14` (not `gcc`, not `afl-clang-fast`)

**Decision:** `libmutator.so` is compiled with `/usr/bin/clang-14`.

**Rationale:**
- The rest of the project (modbus, uart, lava-m targets) uses LLVM-14, so clang-14 is already the pinned toolchain.
- `afl-clang-fast` must **not** be used for `libmutator.so`. It inserts AFL++ SHM instrumentation into the compiled binary; when AFL++ loads the `.so` via `dlopen()`, that instrumentation would double-instrument the fuzzer process itself and corrupt the shared-memory bitmap.
- `gcc` would also work but would produce a `.so` from a different ABI baseline. Using clang-14 throughout eliminates LTO or cross-ABI issues if the project evolves.

---

## D-02 Socket type: `SOCK_SEQPACKET`

**Decision:** The IPC channel uses `AF_UNIX SOCK_SEQPACKET`.

**Rationale:**
- `SOCK_STREAM` (TCP-like): Requires length-prefix framing; susceptible to partial reads under load.
- `SOCK_DGRAM` (UDP-like): Message boundaries preserved but delivery is **not** guaranteed; datagrams can be silently dropped if the receive buffer is full.
- `SOCK_SEQPACKET`: Combines the **message-boundary preservation** of SOCK_DGRAM with the **reliability** of SOCK_STREAM, within the kernel. This means:
  - Each `send()` maps 1:1 to a `recv()` — no framing code needed in either C or Python.
  - Messages are never silently lost; back-pressure is applied instead.
  - Connection-oriented: the C side detects daemon restarts via EOF.

---

## D-03 Batch size: 8 operator IDs per IPC round-trip

**Decision:** `MUTATOR_BATCH_SIZE = 8` (defined in `mutator.h`).

**Rationale:**
- Each `afl_custom_fuzz()` call consumes one operator ID from a local cache.
- If the batch were size 1, every call would incur an IPC round-trip (~5–50 µs for a UNIX socket).
- AFL++ can invoke `afl_custom_fuzz()` tens of thousands of times per second on fast targets.
- Batch size 8 amortizes IPC overhead by 8× while keeping the PPO policy's reaction latency bounded to at most 8 executions (~milliseconds).
- `MUT_BATCH_MAX = 32` in `protocol.h` allows the Python side to send up to 32 IDs per message, giving flexibility to tune batch size at runtime without recompiling.

---

## D-04 Frame cap: `MUTATOR_MAX_SIZE = 256` bytes

**Decision:** The output buffer for `afl_custom_fuzz()` is capped at 256 bytes.

**Rationale:**
- The primary fuzzing target in this project is the Modbus RTU protocol.
- A Modbus RTU frame is at most 256 bytes (1 byte address + 1 function code + 252 data + 2 CRC).
- Allowing larger frames would waste memory and generate inputs that the target parser would immediately reject as oversized.
- Targets with larger frames (e.g., UART async-rx) can override this constant at compile time without changing the mutator logic.

---

## D-05 Fallback policy: random local operator on daemon unavailability

**Decision:** If the IPC socket is unavailable (daemon not started, timeout, or error), `afl_custom_fuzz()` falls back to selecting a random operator from the local C pool.

**Rationale:**
- Prevents fuzzing from halting when the Python PPO daemon is not running (e.g., dry-run testing, CI without daemon).
- The fallback is functionally equivalent to AFL++'s built-in havoc stage, which is already effective. Resilience is preferable to a hard crash.
- The 100 ms `SO_RCVTIMEO` timeout ensures AFL++ is not stalled waiting for a slow daemon response.
- The fallback triggers a reconnect attempt on the next batch boundary, so if the daemon recovers the mutator will automatically reconnect.

---

## D-06 `HISTORY_DIM = 64` vs `TOTAL_OPERATORS = 14`

**Decision:** The `history_vec` in `mut_request_msg` has 64 float entries, even though only 14 operators exist.

**Rationale:**
- The PPO state is 192-dimensional (matching the `ppo_scheduler/config.py` contract):
  - Dimensions 0–127: coverage bitmap projection (16 bytes → 128 bits).
  - Dimensions 128–191: 64-float history vector.
- The 64-float history vector is structured as:
  - `[0:16]`  — normalized operator usage counts (14 used, 2 zero-padded)
  - `[16:32]` — normalized operator success counts
  - `[32:48]` — one-hot encoding of the last operator (14 used, 2 zero-padded)
  - `[48:64]` — campaign telemetry (crashes, hangs, queue size, exec rate)
- Using 64 rather than 42 entries keeps the state vector aligned to a power of 2 and avoids reshape complexity in the Python PPO model.

---

## D-07 Reward formula constants: `LAMBDA_CRASH = 10`, `LAMBDA_NOVELTY = 1`

**Decision:** The deferred reward `R_{t-1}` is computed as:
```
R = LAMBDA_CRASH × Δcrashes + LAMBDA_NOVELTY × Δnovelty_bits
```
where `Δnovelty_bits` is the number of new coverage bits observed since the last snapshot.

**Rationale:**
- Crashes are 10× more valuable than individual new coverage bits because:
  - A single new crash represents a potential vulnerability.
  - New coverage bits are abundant in early fuzzing but become sparse later.
- `LAMBDA_NOVELTY = 1` (per new bit) rather than per-edge is consistent with how AFL++ measures coverage novelty in its bitmap.
- These constants match the `PPOConfig` defaults in `ppo_scheduler/config.py` and must remain synchronized between the C and Python layers.

---

## D-08 `afl_state_t` access pattern in `mutator.c`

**Decision:** The mutator accesses `afl->fsrv.trace_bits`, `afl->fsrv.map_size`, `afl->saved_crashes`, `afl->saved_hangs`, and `afl->current_entry` directly from the live `afl_state_t *` pointer passed to `afl_custom_init()`.

**Rationale:**
- AFL++ custom mutators receive a typed `afl_state_t *` pointer at initialization.
- Direct struct field access is the canonical AFL++ API for custom mutators (see AFL++ docs and example custom mutators in `AFLplusplus/custom_mutators/`).
- The alternative (reading from environment variables or files) would introduce latency and race conditions.
- The AFL++ headers in `../../AFLplusplus/include/` are the pinned source-of-truth for the struct layout, so there is no ABI risk as long as the same AFL++ tree is used for both building `libmutator.so` and running `afl-fuzz`.
