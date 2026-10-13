# Actions & Engineering Decisions Log: DL Seed Optimizer

This document records all implementation actions, engineering design decisions, tools used, hyperparameter values, algorithms, and training/inference flows for the **Deep Learning Seed Optimizer** module.

---

## 1. Overview & Architecture

The DL Seed Optimizer is an autonomous neural network module for seed prioritization in AFL++. It operates independently from the PPO mutation scheduler, evaluating complete ordered execution traces rather than aggregate summary states.

```
AFL++ Execution Trace
    ↓ (via /dev/shm file transport)
Tokenization (edge_id → token_id)
    ↓
Embedding Layer (VOCAB_SIZE=65537, EMBED_DIM=32, padding_idx=0)
    ↓
Single-layer Unidirectional GRU (input=32, hidden=128, no sequence truncation)
    ↓
Latent Vector (128-D)
    ↓
DQN MLP Head: Linear(128 → 64) → ReLU → Linear(64 → 1)
    ↓
Scalar Q(seed) (Descending Q ranking for seed queue prioritization)
```

---

## 2. Tools & Environment

| Tool / Dependency | Version / Path | Purpose |
|-------------------|----------------|---------|
| **OS** | Linux (Ubuntu x86_64) | Development and fuzzing host |
| **Python** | 3.10.12 (`/usr/bin/python3`) | Daemon runtime, ML training & inference |
| **PyTorch** | 2.6.0+cu124 (CUDA 0 active) | Neural network definitions, backprop, autograd |
| **NumPy** | 1.26.4 | Array handling, token conversion, percentiles |
| **pytest** | 9.1.1 | Unit and integration test execution |
| **clang** | 14.0.0 (`/usr/bin/clang-14`) | C compiler for mutator & IPC bridge |
| **AFL++** | 4.08c (`~/Research/AFLplusplus/`) | Target fuzzer and header sources |

---

## 3. Commands Executed

### Build Commands:
```bash
# Compile decoupled IPC bridge client and test harness
make -C /home/captainalan2/Research/adaptive-fuzzer/ipc_bridge clean
make -C /home/captainalan2/Research/adaptive-fuzzer/ipc_bridge all

# Compile C mutator with trace_collector and DL hooks
make -C /home/captainalan2/Research/adaptive-fuzzer/c_mutator clean
make -C /home/captainalan2/Research/adaptive-fuzzer/c_mutator all
```

### Verification & Test Commands:
```bash
# Run IPC bridge unit tests & latency profiler
make -C /home/captainalan2/Research/adaptive-fuzzer/ipc_bridge tests

# Run C aggregator unit tests
/home/captainalan2/Research/adaptive-fuzzer/c_mutator/test_agg

# Run full Python DL module unit test suite
python3 -m pytest dl_seed_optimizer/tests/ -v --tb=short

# Run End-to-End multi-seed simulation test
python3 -m pytest dl_seed_optimizer/tests/test_e2e_integration.py -v -s
```

---

## 4. Hyperparameter Values & Specifications

| Parameter | Value | Source / Justification |
|---|---|---|
| **Vocabulary Size** | `65,537` | AFL++ `MAP_SIZE` (65,536 edges) + 1 (token 0 reserved for padding) |
| **Embedding Dimension** | `32` | Implementation choice; dense embedding table is ~8 MB |
| **GRU Hidden Size** | `128` | Implementation choice; matches latent dimension |
| **GRU Layers** | `1` | Spec §10; single-layer unidirectional GRU |
| **GRU Bidirectional** | `False` | Execution trace ordering is forward-causal |
| **DQN MLP Hidden** | `64` | `Linear(128 → 64) → ReLU → Linear(64 → 1)` |
| **Learning Rate (α)** | `0.001` | **Paper-specified** (spec §27); Adam optimizer |
| **Discount Factor (γ)** | `0.95` | **Paper-specified** (spec §23); TD target computation |
| **Replay Buffer Capacity** | `50,000` | Implementation choice (spec §20); FIFO circular buffer |
| **Batch Size** | `32` | Implementation choice (spec §21); uniform random sampling |
| **Target Network Sync** | `1,000` steps | Hard parameter copy from online to target network |
| **Loss Function** | Huber (`SmoothL1Loss`) | Robust to reward and Q-value outliers |
| **Max Gradient Norm** | `1.0` | Prevents gradient explosion on long traces |
| **Exploration (ε)** | `0.10 → 0.01` | Multiplicative decay (`0.9995` per step) |
| **Acceptance Threshold** | Top 50% percentile | Q-ranking threshold; seeds above median accepted |
| **Trace Truncation** | **None** | Full traces preserved; batching via PyTorch packed sequences |
| **Reward Weights** | `w_crash=10.0, w_nov=1.0` | Matches `aggregator.c` (`LAMBDA_CRASH=10, LAMBDA_NOVELTY=1`) |

---

## 5. Algorithms & Mathematical Formulations

### 5.1 Q-Learning & TD Target
For a transition $(s, r, s', done)$:
$$Q_{\text{target}} = r + \gamma \cdot Q_{\text{target\_net}}(s') \cdot (1 - done)$$
Loss is Huber loss:
$$\mathcal{L} = \text{SmoothL1}(Q_{\text{online}}(s), Q_{\text{target}})$$

### 5.2 Reward Formula
Matching `adaptive-fuzzer/c_mutator/aggregator.c`:
$$R = \text{norm\_cov} + \lambda_{\text{crash}} \cdot I_{\text{crash}} + \lambda_{\text{novelty}} \cdot I_{\text{novel}}$$
$$\text{norm\_cov} = \frac{\Delta\text{cov\_bits}}{\ln(e + \text{total\_execs})}$$

---

## 6. Training & Inference Flows

### Inference Flow (Latency < 2ms):
1. AFL++ mutator hook `afl_custom_queue_get()` triggers for new queue entry `seed_id`.
2. Trace collector dumps execution trace to `/dev/shm/afl_dl_traces/<seed_id>.trace`.
3. Mutator sends `MSG_TRACE_SUBMIT (0x05)` over `/tmp/afl_dl_bridge.sock`.
4. `DLDaemon` reads binary trace file, checks `(seed_id, model_version)` cache.
5. If uncached, tokens are passed through `SeedQNetwork(Embedding → GRU → DQNHead)`.
6. Evaluates acceptance using percentile threshold and $\varepsilon$-greedy exploration.
7. Transmits `MSG_TRACE_SCORE (0x06)` back to mutator with $Q(\text{seed})$ and `accept` byte.

### Training Flow (Asynchronous Background Thread):
1. Fuzzing execution completes; mutator sends `MSG_DL_REWARD (0x07)`.
2. Daemon computes reward and pushes `Transition` to thread-safe `ReplayBuffer`.
3. Background worker continuously polls `ReplayBuffer`.
4. When buffer contains $\ge \text{MIN\_BUFFER\_SIZE}$ items, worker samples batch of 32.
5. `collate_transitions()` dynamically pads sequences.
6. Computes online $Q$ and target $Q$, computes Huber loss, clips gradients, and runs Adam step.
7. Every 1,000 gradient updates, target network weights are synchronized, bumping `model_version` and clearing stale Q-cache entries.

---

## 7. Key Engineering Design Decisions

1. **Decoupled Architecture**: DL module runs as a dedicated daemon on `/tmp/afl_dl_bridge.sock`, leaving PPO on `/tmp/afl_ai_bridge.sock`.
2. **File-Based Trace Transport**: Traces of unbounded length are transferred via tmpfs `/dev/shm` rather than large IPC datagrams.
3. **No Sequence Truncation**: PyTorch `pack_padded_sequence` processes arbitrary sequence lengths without data loss.
4. **Device Agnostic**: Seamless CUDA GPU acceleration with automatic CPU fallback.
5. **Thread Safety**: Mutex locks protect ReplayBuffer and Q-cache between IPC serving and training threads.

---

## 8. Test Verification Summary

- **Unit Tests**: 122/122 passing (`test_tokenizer.py`, `test_models.py`, `test_reward.py`, `test_replay_buffer.py`, `test_trainer.py`, `test_dl_daemon.py`).
- **Integration Tests**: 1/1 passing (`test_e2e_integration.py`), verifying 30 simulated seeds with 3.75 ms average RTT and active experience accumulation.
- **C Native Tests**: `test_agg` 6/6 passing, `test_ipc_client` passing with 9.38 $\mu$s mean RTT.
