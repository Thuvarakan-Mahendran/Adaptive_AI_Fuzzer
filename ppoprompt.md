# PPO Mutation-Strategy Scheduler — Direct Code-Generation Prompt

## 0. Role and objective

Act as a senior Python/PyTorch reinforcement-learning engineer.

Implement the PPO-based mutation-strategy scheduler described below as a clean, testable standalone module.

The implementation must follow the specified 192-dimensional state contract and PPO training procedure. It must be possible to train the PPO agent initially against a deterministic/mock environment without any AFL++ or socket integration.

### Important scope restriction

Do **not** implement any of the following in this task:

- Socket receiver — **defer**
- State serialization — **defer**
- State codec — **defer**
- AFL++ feedback aggregator — **defer**
- UCB/UCB1 action selection — **do not add**
- DQN — **do not add**
- RNN/LSTM/GRU — **do not add**

Keep the PPO scheduler isolated behind a simple environment/state interface. The real AFL++/Unix-socket integration will be added later.

---

# 1. Source-grounded design

The referenced paper defines the RL state as:

- 128-dimensional binary code-coverage representation
- 64-dimensional mutation-history representation
- total state dimension = 192

The paper describes PPO as the RL method and says the action selects a mutation strategy.

The paper also defines reward conceptually as:

\[
R_t =
\Delta coverage
+
\lambda_{crash} I_{crash}
+
\lambda_{novelty} I_{novel}
\]

The paper does **not** specify the exact construction of the 128 coverage bits, the exact partition of the 64 history dimensions, normalization details, or a socket wire format.

Therefore, the following state layout, temporal coverage construction, normalization policy, and software architecture are explicit implementation choices for this project. Do not present them as if they were directly specified by the paper.

---

# 2. Required project structure

Create the following structure:

```text
ppo_scheduler/
├── config.py
├── networks.py
├── buffer.py
├── agent.py
├── losses.py
├── environment.py
├── state_preprocessor.py
├── train.py
├── evaluate.py
├── tests/
│   ├── test_state.py
│   ├── test_networks.py
│   ├── test_buffer.py
│   ├── test_gae.py
│   ├── test_ppo_update.py
│   └── test_environment.py
└── checkpoints/
```

Do not create socket, AFL++, serialization, codec, UCB, DQN, or RNN modules.

If an existing repository already contains equivalent files, modify/reuse them rather than duplicating functionality.

---

# 3. Configuration

Implement a central configuration with these default values:

```python
STATE_DIM = 192
ACTION_DIM = K                  # K is configurable

GAMMA = 0.95
GAE_LAMBDA = 0.95

CLIP_EPSILON = 0.2

ACTOR_LR = 3e-4
CRITIC_LR = 1e-3

ROLLOUT_LENGTH = 128
PPO_EPOCHS = 10
MINIBATCH_SIZE = 32

VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5

LAMBDA_CRASH = 10.0
LAMBDA_NOVELTY = 1.0
```

Make these configurable rather than hard-coding them throughout the implementation.

Validate:

```text
ROLLOUT_LENGTH > 0
PPO_EPOCHS > 0
MINIBATCH_SIZE > 0
ACTION_DIM > 1
STATE_DIM == 192
0 < GAMMA <= 1
0 <= GAE_LAMBDA <= 1
CLIP_EPSILON > 0
```

---

# 4. Exact 192-dimensional state contract

The PPO agent must always receive a tensor shaped:

```text
[batch_size, 192]
```

or for a single state:

```text
[192]
```

Use this exact layout:

```text
state[0:128]     = coverage representation
state[128:144]   = operator usage history
state[144:160]   = operator success history
state[160:176]   = last-operator one-hot
state[176:192]   = campaign telemetry
```

Therefore:

```text
128 + 16 + 16 + 16 + 16 = 192
```

The PPO networks must not receive any other dimensions.

Assume the implementation currently supports up to 16 mutation operators for the 16-dimensional operator-history sections.

If `ACTION_DIM < 16`, the unused operator-history positions must remain zero.

If `ACTION_DIM > 16`, fail validation with a clear error because the agreed state layout reserves only 16 operator slots.

---

# 5. Coverage representation

## 5.1 Required representation

The first 128 dimensions are binary:

```text
coverage[0:128] ∈ {0, 1}
```

These 128 bits represent **recent/differential coverage progress**, not the complete cumulative AFL++ bitmap.

This is an implementation design choice because the paper only specifies a 128-dimensional binary coverage representation.

## 5.2 Temporal behavior

Use a rolling execution window:

```text
W = 100 executions
```

Within the current rolling window, when a newly discovered edge is observed:

```text
bin = hash(edge_id) & 127
coverage[bin] = 1
```

At the end of the window, old coverage discoveries should no longer affect the next window.

The purpose is to avoid the state becoming permanently saturated with ones as cumulative coverage grows.

## 5.3 Hashing

Use a deterministic non-cryptographic hash such as MurmurHash3.

The hash must be deterministic across runs for the same edge identifier.

Do not use Python's built-in `hash()` because its value may vary between processes.

Create a small pure function such as:

```python
def edge_to_bin(edge_id: int) -> int:
    ...
```

with output:

```text
0 <= result < 128
```

## 5.4 Collision behavior

Hash collisions are expected.

If multiple edges map to the same bin, the bin remains:

```text
1
```

No attempt should be made to reconstruct the original edge IDs from the 128-bit state.

## 5.5 Coverage validation

Reject states if any of the first 128 values are not exactly binary.

Allowed:

```text
0
1
```

Reject:

```text
0.5
-1
2
NaN
Inf
```

---

# 6. 64-dimensional mutation-history representation

Use the following exact partition:

```text
128:144   operator usage
144:160   operator success
160:176   last operator
176:192   campaign telemetry
```

All continuous history values must be finite.

---

# 7. Operator usage history

For each of the 16 operator slots maintain an exponentially decayed usage value.

Use:

\[
u_i \leftarrow 0.99u_i + I[a_t=i]
\]

where:

- \(u_i\) = usage value for operator `i`
- \(a_t\) = current action
- `0.99` = decay factor
- \(I[\cdot]\) = indicator function

The `0.99` value is an implementation choice, not a paper-specified hyperparameter.

Normalize the resulting vector using L1 normalization:

\[
\hat u_i =
\frac{u_i}{\sum_j u_j + \epsilon}
\]

with a small epsilon to avoid division by zero.

The resulting values occupy:

```text
state[128:144]
```

---

# 8. Operator success history

Maintain a recent/decayed success estimate for each operator.

A suitable implementation is:

```text
success_i ← decay * success_i + (1 - decay) * success_signal
```

where `success_signal` is:

```text
1 if the selected operator produced the defined success condition
0 otherwise
```

Use the same decay convention consistently throughout the implementation.

Keep values finite and bounded.

Store them in:

```text
state[144:160]
```

The exact success signal should be clearly documented in the mock environment.

Do not invent a second RL objective.

---

# 9. Last-operator representation

Use a single 16-dimensional one-hot vector.

For example, if operator 3 was selected:

```text
state[160:176] =
[0, 0, 0, 1, 0, ..., 0]
```

If there has not yet been an action, use an all-zero vector.

Do **not** implement "last three operators" here.

Three 16-way one-hot vectors would require 48 dimensions and would violate the agreed 64-dimensional history budget.

---

# 10. Campaign telemetry

Use the final 16 dimensions:

```text
state[176:192]
```

Include 16 scalar campaign-level signals.

A recommended set is:

```text
0   queue depth
1   executions since last new coverage
2   normalized execution rate
3   current seed energy
4   elapsed campaign fraction
5   recent crash count
6   recent coverage growth
7   recent reward
8   timeout rate
9   mutation efficiency
10  queue growth
11  time since last crash
12  recent unique-crash rate
13  average execution time
14  seed age
15  reserved/available campaign metric
```

All telemetry values must be normalized into stable numeric ranges before entering the neural network.

Do not allow raw values with uncontrolled magnitudes to reach the network.

If a metric is unavailable in the mock environment, use a deterministic placeholder and document it.

---

# 11. State preprocessing

Create:

```text
state_preprocessor.py
```

This module must be responsible for validating and preparing an already-constructed 192-dimensional state.

It must NOT implement sockets, serialization, or AFL++ integration.

## 11.1 Validation

Implement strict validation:

```python
validate_state(state)
```

Check:

1. shape is exactly `(192,)` for a single state;
2. all values are finite;
3. coverage section is binary;
4. last-operator section is either all zero or valid one-hot;
5. operator history dimensions are finite;
6. telemetry dimensions are finite;
7. action/operator indices are within the supported range.

Raise informative exceptions rather than silently correcting malformed states.

## 11.2 Normalization

Do not independently standardize each individual state.

Do not transform the binary coverage representation.

Coverage must remain:

```text
0 / 1
```

Normalize only the continuous history/telemetry features.

Use fixed or running normalization statistics consistently across training and evaluation.

The normalization statistics must be saveable/restorable with checkpoints.

Avoid data leakage between evaluation and training.

---

# 12. Neural networks

## 12.1 Actor

Implement:

```text
192
 ↓
256
 ↓
256
 ↓
ACTION_DIM
```

Use standard fully connected layers with nonlinear activations.

The actor output must represent action logits.

Use a categorical distribution:

```python
Categorical(logits=logits)
```

Do not use:

- recurrent layers;
- attention;
- DQN;
- Q-values;
- UCB;
- separate policy heads.

## 12.2 Critic

Implement:

```text
192
 ↓
256
 ↓
256
 ↓
1
```

Output a scalar state-value estimate.

The critic must estimate:

\[
V(s_t)
\]

not Q-values.

---

# 13. PPO action selection

Implement an agent class responsible for:

```python
select_action(state)
```

Given the validated 192-dimensional state:

1. preprocess it;
2. convert to `torch.float32`;
3. add batch dimension if necessary;
4. pass it through the actor;
5. construct a categorical distribution;
6. sample an action;
7. compute its log probability;
8. compute the critic value;
9. return:
   - action
   - log probability
   - value
   - optionally processed state

Do not use deterministic argmax during training.

Exploration comes from sampling the categorical policy.

For evaluation, provide an optional deterministic mode using the highest-probability action.

---

# 14. Rollout buffer

Implement a rollout buffer storing at least:

```text
states
actions
rewards
dones
values
old_log_probs
next_states
```

The buffer length must support:

```text
ROLLOUT_LENGTH = 128
```

At 128 transitions:

1. stop collecting;
2. freeze the rollout;
3. calculate advantages;
4. calculate returns;
5. perform PPO updates;
6. clear the buffer;
7. resume collection.

Do not update the policy halfway through the rollout.

---

# 15. Reward handling

Implement the reward formula:

\[
R_t =
\Delta coverage
+
\lambda_{crash} I_{crash}
+
\lambda_{novelty} I_{novel}
\]

with:

```text
LAMBDA_CRASH   = 10.0
LAMBDA_NOVELTY = 1.0
```

`Δcoverage` must represent the coverage improvement attributable to the current transition.

The mock environment must make the three reward components inspectable for testing.

Do not add extra reward terms unless explicitly configured.

---

# 16. GAE

Implement Generalized Advantage Estimation:

\[
\delta_t =
r_t + \gamma V(s_{t+1}) (1-d_t) - V(s_t)
\]

and:

\[
A_t =
\delta_t +
\gamma\lambda(1-d_t)A_{t+1}
\]

Use:

```text
gamma = 0.95
lambda = 0.95
```

Compute returns:

\[
R_t = A_t + V(s_t)
\]

Normalize advantages before PPO optimization:

```text
advantages = (advantages - mean) / (std + epsilon)
```

Test the GAE implementation independently.

---

# 17. PPO loss

For each minibatch calculate:

```text
ratio = exp(new_log_prob - old_log_prob)
```

Then:

\[
L^{CLIP}
=
-\mathbb{E}
[
\min(
r_t A_t,
clip(r_t,1-\epsilon,1+\epsilon)A_t
)
]
\]

with:

```text
epsilon = 0.2
```

Critic loss:

```text
value_loss = MSE(predicted_value, returns)
```

Entropy:

```text
entropy = distribution.entropy().mean()
```

Combined objective:

```text
total_loss =
policy_loss
+ VALUE_COEF * value_loss
- ENTROPY_COEF * entropy
```

Use:

```text
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
```

Clip gradients:

```text
MAX_GRAD_NORM = 0.5
```

---

# 18. PPO update schedule

For every completed rollout:

```text
10 PPO epochs
×
minibatches of 32
```

Therefore:

```text
PPO_EPOCHS = 10
MINIBATCH_SIZE = 32
```

Shuffle minibatch indices each epoch.

For every minibatch:

1. evaluate actor;
2. compute new log probabilities;
3. compute entropy;
4. evaluate critic;
5. calculate PPO clipped policy loss;
6. calculate value loss;
7. combine losses;
8. zero gradients;
9. backpropagate;
10. clip gradients;
11. optimizer step.

Do not recompute or replace the stored old log probabilities during the PPO update.

The old policy values must remain frozen for the entire rollout update.

---

# 19. Separate actor and critic optimizers

Use:

```text
Actor learning rate  = 3e-4
Critic learning rate = 1e-3
```

Create separate optimizers.

Do not share optimizer state between actor and critic.

---

# 20. Mock environment

Create:

```text
environment.py
```

Implement a deterministic/testable mock mutation environment.

The environment should:

1. expose `reset()`;
2. expose `step(action)`;
3. return a valid 192-dimensional state;
4. maintain mutation-operator statistics;
5. simulate coverage progress;
6. simulate crash events;
7. simulate novelty events;
8. calculate the specified reward;
9. expose enough information for tests.

The environment is only a stand-in for the future AFL++ feedback path.

Do not implement actual AFL++ execution.

Do not implement Unix sockets.

Do not implement state serialization.

---

# 21. Environment/action semantics

The action is:

```text
action ∈ {0, ..., ACTION_DIM - 1}
```

It represents a mutation strategy.

The environment must update:

```text
operator usage
operator success
last operator
coverage state
telemetry
```

after every action.

The next observation must still have exactly 192 dimensions.

The environment should make it possible to verify that different mutation strategies can receive different simulated outcomes.

---

# 22. Training loop

Create:

```text
train.py
```

The training loop must follow:

```text
initialize environment
initialize PPO agent

while training:
    state = env.reset()

    for step in rollout:
        action, log_prob, value = agent.select_action(state)

        next_state, reward, done, info = env.step(action)

        buffer.add(
            state,
            action,
            reward,
            done,
            value,
            log_prob,
            next_state
        )

        state = next_state

        if done:
            state = env.reset()

        if buffer.is_full():
            agent.update(buffer)
            buffer.clear()
```

Handle bootstrap value correctly when the rollout ends because the 128-step collection boundary is reached without an environment terminal state.

Do not confuse:

```text
rollout boundary
```

with:

```text
environment terminal state
```

---

# 23. Checkpointing

Save:

```text
actor state_dict
critic state_dict
actor optimizer state
critic optimizer state
configuration
normalization statistics
training step
```

Use a versioned checkpoint format.

Provide loading functionality that restores the complete training state.

Reject incompatible checkpoints when:

```text
STATE_DIM != 192
```

or the action dimension is incompatible.

---

# 24. Evaluation

Create:

```text
evaluate.py
```

Evaluation must:

1. load a checkpoint;
2. restore normalization statistics;
3. create the same 192-dimensional state interface;
4. run without training updates;
5. support deterministic action selection;
6. report at least:
   - total reward;
   - mean reward;
   - coverage progress;
   - crash count;
   - novelty count;
   - action-selection frequencies.

Do not update normalization statistics during evaluation.

---

# 25. Unit tests

Implement tests for all critical interfaces.

## 25.1 State tests

Verify:

```text
state.shape == (192,)
```

Verify:

```text
state[0:128] are binary
```

Verify:

```text
state[128:144] has length 16
state[144:160] has length 16
state[160:176] has length 16
state[176:192] has length 16
```

Reject:

```text
191-dimensional states
193-dimensional states
NaN
Inf
non-binary coverage
invalid last-operator vectors
```

## 25.2 Coverage tests

Verify:

1. same edge always maps to same bin;
2. bin is in `[0,127]`;
3. collisions are handled;
4. rolling-window behavior removes stale coverage after the configured window;
5. cumulative coverage does not permanently force all 128 bits to one.

## 25.3 Network tests

Verify:

```text
actor(torch.zeros(1, 192))
```

returns:

```text
[1, ACTION_DIM]
```

and:

```text
critic(torch.zeros(1, 192))
```

returns:

```text
[1, 1]
```

Verify invalid state dimensions fail clearly.

## 25.4 Buffer tests

Verify:

1. transitions are stored;
2. buffer detects full rollout;
3. old log probabilities remain unchanged;
4. GAE outputs correct shapes;
5. returns and advantages contain finite values;
6. buffer clears correctly.

## 25.5 PPO tests

Verify:

1. action sampling works;
2. log probabilities are finite;
3. values are finite;
4. PPO update changes trainable parameters;
5. entropy is finite;
6. clipped objective is finite;
7. gradients are clipped;
8. no NaNs are introduced.

## 25.6 Environment tests

Verify:

```text
reset() -> valid 192-D state
step(action) -> valid next state
```

for every supported action.

Verify reward components match:

```text
Δcoverage
+ crash bonus
+ novelty bonus
```

---

# 26. Reproducibility

Provide a seed function that seeds:

```text
Python random
NumPy
PyTorch
```

Use deterministic behavior where practical.

The same seed should produce repeatable mock-environment behavior and model initialization.

Document any remaining nondeterminism.

---

# 27. Device handling

Support:

```text
CPU
CUDA when available
```

Do not assume CUDA exists.

Select device through configuration or automatic detection.

Ensure:

```text
state tensors
actor
critic
```

are on the same device.

---

# 28. Logging

Training should report periodically:

```text
training step
rollout reward
mean reward
policy loss
value loss
entropy
mean advantage
mean value
coverage growth
crash count
novelty count
action distribution
```

Do not flood stdout every environment step.

Use a configurable logging interval.

---

# 29. Numerical stability

Protect against:

```text
NaN
Inf
division by zero
zero-variance advantage normalization
invalid categorical distributions
```

Use small epsilons where appropriate.

Before every optimizer update, optionally assert/debug-check that losses are finite.

Fail early with useful diagnostics when non-finite values occur.

---

# 30. Architecture boundary

The final data flow for this implementation must be:

```text
Mock Environment
       │
       ▼
192-D State
       │
       ▼
State Validation / Preprocessing
       │
       ▼
float32 Tensor [192]
       │
       ├──────────────► Actor ─────► action distribution
       │
       └──────────────► Critic ────► V(s)
                         │
                         ▼
                    PPO Training
```

The future production pipeline will eventually replace the mock environment boundary with an external feedback path, but that is outside this task.

---

# 31. Do not add these components

The following are explicitly forbidden in this implementation:

## Socket receiver

Do not create:

```text
socket_client.py
unix_socket.py
receiver.py
```

No socket communication.

## State serialization

Do not implement binary packing/unpacking.

Do not define a wire protocol.

## State codec

Do not implement conversion from AFL++ binary structures.

The current implementation operates directly on Python/NumPy/PyTorch state vectors.

## AFL++ aggregator

Do not implement AFL++ hooks, external mutators, shared memory, forkserver integration, or AFL++ bitmap processing.

## UCB

Do not implement:

```text
UCB
UCB1
upper confidence bound
bandit selection
```

PPO's categorical policy is the sole action-selection mechanism.

## DQN

Do not create:

```text
Q-network
target network
replay buffer
epsilon-greedy
Q-learning
```

## RNN

Do not create:

```text
LSTM
GRU
RNN
recurrent policy
recurrent critic
```

The current policy and critic are feed-forward networks.

---

# 32. Important implementation distinction

Do not silently change the agreed state representation to match assumptions from AFL++.

The PPO model does **not** need to know:

- AFL++ bitmap size;
- raw edge IDs;
- raw execution traces;
- socket message layout;
- mutation implementation details.

It only receives:

```text
192 numerical features
```

with the exact layout defined above.

The PPO implementation should therefore remain independent of AFL++.

---

# 33. Code quality requirements

Use:

- Python 3.10+
- PyTorch
- NumPy
- pytest
- type hints where useful
- docstrings for public classes/functions
- clear separation of concerns

Avoid:

- global mutable state;
- duplicated PPO math;
- hidden normalization;
- magic numbers;
- unnecessary abstractions;
- dead code for deferred integrations.

Every configurable value should come from the configuration.

---

# 34. Expected public interfaces

Implement interfaces approximately like:

```python
class PPOAgent:
    def select_action(self, state, deterministic=False):
        ...

    def update(self, rollout_buffer):
        ...

    def save(self, path):
        ...

    def load(self, path):
        ...
```

```python
class RolloutBuffer:
    def add(self, state, action, reward, done, value, log_prob, next_state):
        ...

    def compute_returns_and_advantages(self, next_value, gamma, gae_lambda):
        ...

    def get_minibatches(self, batch_size):
        ...

    def clear(self):
        ...
```

```python
class StatePreprocessor:
    def validate(self, state):
        ...

    def normalize(self, state):
        ...

    def process(self, state):
        ...
```

```python
class MockMutationEnvironment:
    def reset(self):
        ...

    def step(self, action):
        ...
```

Adapt exact signatures if necessary, but preserve the responsibilities.

---

# 35. Implementation order

Implement in this exact order:

### Step 1 — Configuration

Create `config.py`.

Implement all PPO/state constants and validation.

### Step 2 — State representation

Implement the 192-D state layout and validation.

### Step 3 — State preprocessing

Implement continuous-feature normalization while preserving binary coverage.

### Step 4 — Coverage helper

Implement deterministic edge hashing and rolling-window coverage behavior.

### Step 5 — Neural networks

Implement actor and critic.

### Step 6 — Rollout buffer

Implement storage, GAE, returns, batching, and clearing.

### Step 7 — PPO losses

Implement clipped policy loss, value loss, entropy, and combined loss.

### Step 8 — PPO agent

Implement action selection, rollout update, optimizers, gradient clipping, checkpointing.

### Step 9 — Mock environment

Implement mutation actions, simulated feedback, state construction, and reward.

### Step 10 — Training script

Connect environment → agent → rollout buffer → PPO updates.

### Step 11 — Evaluation script

Implement checkpoint loading and deterministic evaluation.

### Step 12 — Tests

Implement and run all unit/integration tests.

### Step 13 — Smoke test

Run a short training session and verify:

- no shape errors;
- no NaNs;
- PPO parameters change;
- rewards are finite;
- state always remains 192-D;
- action IDs remain valid.

---

# 36. Acceptance criteria

The implementation is complete only when all of the following are true:

- [ ] PPO input dimension is exactly 192.
- [ ] Coverage occupies dimensions 0–127.
- [ ] Operator usage occupies 128–143.
- [ ] Operator success occupies 144–159.
- [ ] Last operator occupies 160–175.
- [ ] Telemetry occupies 176–191.
- [ ] Coverage values are binary.
- [ ] Continuous features are normalized consistently.
- [ ] Actor is `192 → 256 → 256 → ACTION_DIM`.
- [ ] Critic is `192 → 256 → 256 → 1`.
- [ ] Actor uses categorical action sampling.
- [ ] PPO uses clipped policy objective with epsilon 0.2.
- [ ] GAE uses gamma 0.95 and lambda 0.95.
- [ ] Rollouts contain 128 interactions.
- [ ] PPO uses 10 epochs.
- [ ] Minibatch size is 32.
- [ ] Actor LR is 3e-4.
- [ ] Critic LR is 1e-3.
- [ ] Value coefficient is 0.5.
- [ ] Entropy coefficient is 0.01.
- [ ] Gradient norm is clipped at 0.5.
- [ ] Crash reward coefficient is 10.0.
- [ ] Novelty reward coefficient is 1.0.
- [ ] Checkpoints include normalization statistics.
- [ ] Evaluation restores normalization statistics.
- [ ] Tests cover state validation, GAE, PPO updates, and environment behavior.
- [ ] No socket code exists.
- [ ] No serialization/codec code exists.
- [ ] No AFL++ aggregator exists.
- [ ] No UCB exists.
- [ ] No DQN exists.
- [ ] No RNN exists.

---

# 37. Final instruction to the coding agent

Implement the system completely rather than merely describing it.

Before finishing:

1. inspect the existing repository structure;
2. reuse compatible existing PPO code where possible;
3. make only the changes necessary to satisfy this specification;
4. run the test suite;
5. run a short end-to-end mock training smoke test;
6. report files created/modified;
7. report test results;
8. report any implementation choice that could not be satisfied exactly.

Do not implement deferred integrations.

The final system must be a clean standalone PPO mutation-strategy scheduler whose only required external interface at this stage is a validated 192-dimensional state vector.

# 38. Create actions.md

create a markdown file, providing what has been developed such as functions, hyper parameters used and the reasons for such value, why we need those hyperparameters, process flow and the commands used in the right order to create this module, dependancies and tools installed and the way the module was tested and the future works to be done to integrate. To grasp idea on the actions that have been made.
