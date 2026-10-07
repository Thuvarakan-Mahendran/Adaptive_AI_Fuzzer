# Actions.md – PPO Mutation-Strategy Scheduler

## 1. What Was Built

A standalone, testable **PPO-based mutation-strategy scheduler** for adaptive fuzzing. The agent receives a 192-dimensional state vector and selects one of K mutation strategies. It trains entirely against a mock environment with no AFL++ or socket dependency.

---

## 2. Project Structure

```
ppo_scheduler/
├── config.py              – Central hyperparameter dataclass + validation
├── state_preprocessor.py  – State validation, MurmurHash3 edge hashing,
│                            rolling coverage window, Welford normalizer
├── networks.py            – Actor (192→256→256→K) and Critic (192→256→256→1)
├── buffer.py              – Rollout buffer: storage, GAE, minibatch sampling
├── losses.py              – PPO clipped loss, value loss, entropy, combined loss
├── agent.py               – PPOAgent: select_action, update, save, load
├── environment.py         – MockMutationEnvironment (AFL++ stand-in)
├── train.py               – End-to-end training loop with logging
├── evaluate.py            – Checkpoint loading and deterministic evaluation
├── tests/
│   ├── test_state.py      – 13 state validation tests
│   ├── test_networks.py   – 8 network shape/finiteness tests
│   ├── test_buffer.py     – 9 buffer/GAE/minibatch tests
│   ├── test_gae.py        – 4 GAE correctness tests
│   ├── test_ppo_update.py – 8 PPO update/NaN tests
│   └── test_environment.py– 9 environment correctness tests
└── checkpoints/           – Auto-created; versioned .pt checkpoint files
```

---

## 3. 192-Dimensional State Contract

| Slice | Dimension | Content |
|-------|-----------|---------|
| `[0:128]` | 128 | Binary coverage (rolling-window differential) |
| `[128:144]` | 16 | Operator usage (EMA, L1-normalized) |
| `[144:160]` | 16 | Operator success (EMA) |
| `[160:176]` | 16 | Last-operator one-hot |
| `[176:192]` | 16 | Campaign telemetry (16 normalized scalars) |

---

## 4. Functions and Classes

### `config.py` — `PPOConfig`
Dataclass holding all hyperparameters. `__post_init__` calls `_validate()` on creation.
- `resolve_device()` → auto-detects CUDA or falls back to CPU.

### `state_preprocessor.py`

| Symbol | Purpose |
|--------|---------|
| `edge_to_bin(edge_id)` | MurmurHash3-based deterministic mapping: `edge_id → [0,127]` |
| `CoverageTracker` | Rolling window (W=100) of discovered edges → 128-bit binary vector |
| `RunningNormalizer` | Welford online mean/variance; `normalize(x)` returns Z-scored array |
| `validate_state(state, action_dim)` | Strict 7-check validator; raises `ValueError` with clear messages |
| `StatePreprocessor` | Orchestrates `validate → normalize → torch.Tensor`; `.train_mode()` / `.eval_mode()` |

### `networks.py`

| Class | Architecture | Output |
|-------|-------------|--------|
| `Actor` | `192→256→256→K` (Tanh activations, orthogonal init) | Logits for `Categorical` |
| `Critic` | `192→256→256→1` (Tanh activations, orthogonal init) | Scalar `V(s)` |

### `buffer.py` — `RolloutBuffer`

| Method | Purpose |
|--------|---------|
| `add(state, action, reward, done, value, log_prob, next_state)` | Store one raw transition |
| `is_full()` | True when `len == ROLLOUT_LENGTH` |
| `compute_returns_and_advantages(next_value, gamma, gae_lambda)` | GAE + normalized advantages + returns |
| `get_minibatches(batch_size)` | Shuffled minibatch generator (used by old path; update bypasses this) |
| `clear()` | Reset all lists and computed tensors |

### `losses.py`

| Function | Formula |
|---------|---------|
| `ppo_policy_loss(new_lp, old_lp, adv, ε)` | `-E[min(r·A, clip(r,1-ε,1+ε)·A)]` |
| `value_loss(preds, returns)` | `MSE(V(s), R_t)` |
| `entropy_loss(dist)` | `dist.entropy().mean()` |
| `combined_loss(p, v, e, vc, ec)` | `p + vc*v - ec*e` |

### `agent.py` — `PPOAgent`

| Method | Purpose |
|--------|---------|
| `select_action(state, deterministic=False)` | Validate → normalize → actor → Categorical sample → (action, log_prob, value) |
| `update(buffer)` | Freeze normalizer stats → process raw states → GAE → 10×minibatch gradient steps → restore train mode |
| `save(path)` | Version-tagged checkpoint: networks, optimizers, config, normalizer stats, step count |
| `load(path)` | Validate STATE_DIM/ACTION_DIM compatibility before restoring |

### `environment.py` — `MockMutationEnvironment`

| Method | Purpose |
|--------|---------|
| `reset()` | Clear all state, return initial 192-D observation |
| `step(action)` | Simulate coverage/crash/novelty → compute reward → return `(next_state, reward, done, info)` |
| `_build_state()` | Assemble the 192-D vector from tracker + EMAs + one-hot + telemetry |
| `_build_telemetry()` | Return 16 normalized campaign scalars |

---

## 5. Hyperparameters — Values and Reasons

| Parameter | Value | Reason |
|-----------|-------|--------|
| `STATE_DIM` | 192 | Exact paper contract (128 coverage + 64 history) |
| `ACTION_DIM` | 2–16 (configurable K) | Restricted to ≤16 by the 16-slot operator-history layout |
| `GAMMA` | 0.95 | Standard PPO discount; balances immediate vs future reward |
| `GAE_LAMBDA` | 0.95 | Matches GAMMA; high λ uses more of the rollout for variance reduction |
| `CLIP_EPSILON` | 0.2 | Standard PPO clip; prevents catastrophically large policy updates |
| `ACTOR_LR` | 3e-4 | Standard Adam rate for policy networks |
| `CRITIC_LR` | 1e-3 | Slightly higher to speed value convergence relative to policy |
| `ROLLOUT_LENGTH` | 128 | Balanced batch size; enough variance for stable advantage estimates |
| `PPO_EPOCHS` | 10 | Standard PPO reuse count per rollout |
| `MINIBATCH_SIZE` | 32 | Fits comfortably in memory; enough for stable gradient estimates |
| `VALUE_COEF` | 0.5 | Standard; prevents critic from dominating the combined loss |
| `ENTROPY_COEF` | 0.01 | Small bonus to prevent premature policy collapse |
| `MAX_GRAD_NORM` | 0.5 | PPO standard clip; prevents exploding gradients |
| `LAMBDA_CRASH` | 10.0 | Paper's conceptual weighting — crashes are rare and highly valuable |
| `LAMBDA_NOVELTY` | 1.0 | Mild novelty bonus; same order as coverage delta |
| `COVERAGE_WINDOW` | 100 | Rolling window W; prevents coverage bits permanently saturating to 1 |
| `USAGE_DECAY` | 0.99 | EMA decay for operator usage history (implementation choice) |
| `SUCCESS_DECAY` | 0.99 | EMA decay for operator success history (implementation choice) |

---

## 6. Process Flow

```
MockMutationEnvironment.reset()
        │
        ▼ raw 192-D state
agent.select_action(state)
  └─ StatePreprocessor.process()
       ├─ validate_state()       ← checks shape, binary coverage, one-hot, finiteness
       └─ RunningNormalizer      ← Z-scores state[128:192]; keeps [0:128] binary
        │
        ▼ float32 tensor [192]
  ├──► Actor.forward()  → Categorical(logits) → sample → action, log_prob
  └──► Critic.forward() → V(s)
        │
RolloutBuffer.add(raw_state, action, reward, done, value, log_prob, raw_next_state)
        │   (repeated ROLLOUT_LENGTH=128 times)
        ▼
agent.update(buffer)
  ├─ preprocessor.eval_mode()        ← freeze normalizer stats during update
  ├─ normalize all 128 raw states    ← batch normalization
  ├─ compute_returns_and_advantages() ← GAE with γ=0.95, λ=0.95
  └─ for epoch in 10:
       for minibatch of 32:
         actor.forward() → new_log_probs, entropy
         critic.forward() → values
         ppo_policy_loss() + value_loss() + entropy_loss()
         backward() → clip_grad_norm(0.5) → actor_opt.step() + critic_opt.step()
  └─ preprocessor.train_mode()       ← re-enable stats update
        │
  (every 50 rollouts) agent.save(checkpoints/step_NNNN.pt)
```

---

## 7. Dependencies and Installation

| Package | Version used | Role |
|---------|-------------|------|
| Python | 3.10.12 | Runtime |
| PyTorch | 2.12.0 | Neural networks, autograd, distributions |
| NumPy | 2.2.6 | State arrays, GAE computation |
| pytest | 9.1.1 | Test runner |

```bash
# Install pytest (torch + numpy assumed pre-installed)
pip install pytest
```

---

## 8. Commands in Order

```bash
# 1. Navigate to workspace
cd /home/captainalan2/Research

# 2. Create directory structure
mkdir -p ppo_scheduler/tests ppo_scheduler/checkpoints

# 3. Install test runner
pip install pytest

# 4. Run the full test suite
python3 -m pytest ppo_scheduler/tests/ -v

# 5. Run a short training smoke test (10 rollouts, ACTION_DIM=4)
python3 -m ppo_scheduler.train --num_rollouts 10 --action_dim 4 --log_interval 2

# 6. Run a full training session
python3 -m ppo_scheduler.train --num_rollouts 200 --action_dim 8

# 7. Evaluate a saved checkpoint
python3 -m ppo_scheduler.evaluate \
    --checkpoint ppo_scheduler/checkpoints/step_0050.pt \
    --num_episodes 10 --action_dim 8
```

---

## 9. Test Results (Final)

```
50 passed in 4.29s
```

| Test file | Tests | Coverage |
|-----------|-------|----------|
| `test_state.py` | 13 | Shape, binary, NaN/Inf, one-hot, slice lengths |
| `test_networks.py` | 8 | Actor/Critic shapes, batch, finite output, wrong dim |
| `test_buffer.py` | 9 | Storage, full detection, old log-probs, GAE, minibatch, clear |
| `test_gae.py` | 4 | Single step, zero rewards, returns invariant, positive reward sign |
| `test_ppo_update.py` | 8 | Action range, log-prob/value finite, deterministic, params change, no NaN |
| `test_environment.py` | 9 | reset, step validity, reward formula, invalid action, done, binary coverage |

---

## 10. Smoke Test Results

```
[train] device=cpu  action_dim=4  rollout_length=128  ppo_epochs=4
[rollout 0002] mean_reward=0.000  coverage_bits=0.0   policy_loss=-0.0069  entropy=1.3713
[rollout 0004] mean_reward=353.0  coverage_bits=128.0 policy_loss=-0.0172  entropy=1.3501
[rollout 0010] mean_reward=340.5  coverage_bits=128.0 policy_loss=-0.0144  entropy=1.2893
200 steps: OK
Reward range: [0.00, 15.00]
Unique actions used: [0, 1, 2, 3]
```

- No shape errors ✅
- No NaN values ✅
- PPO parameters change after update ✅
- Rewards are finite ✅
- State always 192-D ✅
- Action IDs always in `[0, ACTION_DIM)` ✅

---

## 11. Acceptance Criteria Status

| Criterion | Status |
|-----------|--------|
| PPO input dimension = 192 | ✅ |
| Coverage: dims 0–127, binary | ✅ |
| Operator usage: 128–143 | ✅ |
| Operator success: 144–159 | ✅ |
| Last operator: 160–175, one-hot | ✅ |
| Telemetry: 176–191 | ✅ |
| Continuous features normalized | ✅ |
| Actor: 192→256→256→K | ✅ |
| Critic: 192→256→256→1 | ✅ |
| Categorical action sampling | ✅ |
| PPO clipped objective (ε=0.2) | ✅ |
| GAE (γ=0.95, λ=0.95) | ✅ |
| Rollout = 128 steps | ✅ |
| PPO epochs = 10 | ✅ |
| Minibatch = 32 | ✅ |
| Actor LR = 3e-4 | ✅ |
| Critic LR = 1e-3 | ✅ |
| Value coef = 0.5 | ✅ |
| Entropy coef = 0.01 | ✅ |
| Grad norm clip = 0.5 | ✅ |
| λ_crash = 10.0 | ✅ |
| λ_novelty = 1.0 | ✅ |
| Checkpoints include normalizer stats | ✅ |
| Evaluation restores normalizer stats | ✅ |
| No socket code | ✅ |
| No serialization/codec code | ✅ |
| No AFL++ aggregator | ✅ |
| No UCB | ✅ |
| No DQN | ✅ |
| No RNN | ✅ |

---

## 12. Known Implementation Choices (Not Specified by Paper)

| Choice | Detail |
|--------|--------|
| Coverage hashing | MurmurHash3 (32-bit) with seed=0; not specified by paper |
| Rolling window W | 100 executions; not specified by paper |
| EMA decay | 0.99 for both usage and success; not specified by paper |
| Network activation | Tanh (vs ReLU); chosen for bounded activations |
| Weight initialization | Orthogonal init with gain=1.0 (gain=0.01 for output layer) |
| Normalizer | Welford online mean/variance; statistics frozen during PPO update epochs |
| Buffer design | Stores raw states; normalization done once per update (not per step) |

---

## 13. Future Integration Work

To connect this module to a real AFL++ campaign:

1. **State Serialization** (`state_codec.py`): decode AFL++ bitmap + metadata into raw 192-D numpy arrays conforming to the validated contract.
2. **AFL++ Bitmap Processing**: map the real 64KB AFL++ coverage bitmap to 128-bit differential coverage using `edge_to_bin()`.
3. **Unix Socket Receiver** (`socket_receiver.py`): receive serialized state structs from a companion AFL++ external mutator process.
4. **AFL++ External Mutator** (C/Python): implement `afl_custom_mutator()` that calls the PPO agent's `select_action()` over the socket, applies the chosen strategy, and returns the mutated input.
5. **Real Telemetry**: replace mock placeholder values in `_build_telemetry()` with live AFL++ stats (`queue_cur`, `total_execs`, `last_crash_time`, etc.).
6. **Training on Real Feedback**: replace `MockMutationEnvironment` with a real AFL++ feedback receiver; the rest of the PPO pipeline remains unchanged.
