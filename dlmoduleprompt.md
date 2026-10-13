Yes. Based on the context you added and the decisions we have now made, I would implement the **DL seed optimizer as a separate learning module from PPO**, with the paper's core flow preserved and with the earlier unnecessary additions removed.

The most important final decisions are:

1. **PPO's 192-D state is NOT an input to the DL seed optimizer.**
2. **The DL optimizer input is the execution trace of the individual seed.**
3. **We do not impose the previously proposed 256-event bound.** The full ordered trace is retained.
4. **The trace is collected in the AFL++ instrumentation/feedback layer**, not by modifying the target program. 
5. **RNN = embedding + single GRU** as our concrete implementation.
6. **GRU output = latent representation of that seed's execution trace.**
7. **DQN receives that latent representation and outputs one scalar utility/Q-value.**
8. **The DQN's Q-learning training uses the reward obtained after fuzzing the seed.**
9. **Replay buffer + target network + Huber loss are implementation stabilizations**, not claims about what the paper explicitly specifies.
10. **The final output of inference is one scalar per seed**, which is then used for prioritization.

The paper itself explicitly gives the core architecture as execution trace → RNN → latent vector → DQN, with Q-learning parameters \(\alpha=0.001\), \(\gamma=0.95\), and seed ranking based on predicted utility. 

---

# 1. Final architecture we should implement

The clean version is:

```text
                         AFL++
                           │
                           ▼
                     Seed / Input
                           │
                           ▼
                    Target execution
                           │
                           ▼
              ┌────────────────────────┐
              │  Full ordered execution │
              │         trace           │
              └────────────┬───────────┘
                           │
                           ▼
                   Trace serialization
                           │
                           ▼
                    Trace token IDs
                           │
                           ▼
                  Trainable embedding
                           │
                           ▼
                     Single GRU
                           │
                           ▼
                 Latent trace vector
                           │
                           ▼
                       DQN MLP
                           │
                           ▼
                   scalar Q-value
                           │
                           ▼
                    Seed ranking
                           │
                           ▼
                  AFL++ prioritization
```

This is much closer to what the paper actually says than our earlier architecture containing:

```text
GRU latent + 64-D campaign state
```

The paper does **not** specify that additional campaign-state vector for the DL seed optimizer. It does explicitly specify the 128-D coverage + 64-D mutation-history state for **PPO**, which is a separate module. 

---

# 2. Keep PPO and DL optimizer completely separate

This should be the first architectural rule in your implementation.

## PPO

```text
AFL++ feedback
     │
     ▼
128-D coverage
     +
64-D mutation history
     │
     ▼
192-D PPO state
     │
     ▼
PPO
     │
     ▼
mutation strategy
```

The paper explicitly defines this 192-D PPO state. 

## DL seed optimizer

```text
Seed
 │
 ▼
execution
 │
 ▼
ordered execution trace
 │
 ▼
RNN
 │
 ▼
latent vector
 │
 ▼
DQN
 │
 ▼
Q(seed)
```

The paper explicitly describes this second pathway. 

**Do not merge these representations.**

---

# 3. Step 1 — Execute a seed

Suppose AFL++ has:

```text
seed_0421
```

AFL++ executes it against the target.

The target execution produces the normal AFL++ feedback:

```text
coverage
crash/fault information
execution result
```

But for the DL optimizer, we additionally need:

```text
ordered execution trace
```

The paper describes the feedback aggregator as consolidating runtime coverage, execution traces and fault metadata for the learning modules. 

---

# 4. Step 2 — Collect the execution trace

This is where the trace collector belongs.

**Do not put trace-recording logic inside the target application's source code.**

For your AFL++ + QEMU setup, the collector should operate alongside the AFL++ instrumentation/execution mechanism where basic-block/edge transitions are already observable. 

Conceptually:

```text
                 QEMU / AFL++ execution
                         │
                ┌────────┴────────┐
                │                 │
                ▼                 ▼
        AFL++ coverage       ordered trace
          bitmap             e1,e2,e3,...
                │                 │
                │                 ▼
                │             DL optimizer
                ▼
             PPO
```

---

# 5. What exactly is the trace?

This is an important implementation decision.

We previously considered:

```text
(edge_id, hit_bucket, fault_flags)
```

and a 256-event limit.

**I would not lock that in anymore.**

The paper only requires an execution trace representing temporal coverage/fault feedback. It does not define the exact serialization. 

Since you have decided not to discard events, the implementation should retain the **complete ordered sequence** produced during that seed execution.

For example:

```text
Trace(seed_0421) =
[
    e_1,
    e_2,
    e_3,
    e_4,
    ...
    e_N
]
```

where:

```text
N = actual number of recorded trace events
```

and **N is not artificially limited to 256**.

---

# 6. What should one trace event contain?

Here I recommend a staged approach.

### Minimum version

Start with:

```text
event = edge_id
```

Therefore:

```text
Trace =
[
    edge_12,
    edge_18,
    edge_27,
    edge_31,
    edge_18,
    ...
]
```

This is the cleanest implementation because the RNN is primarily learning the **temporal sequence of execution transitions**.

### Later extension

If your instrumentation already provides reliable hit-count/fault information at event level, you can extend the token:

```text
event =
(edge_id, hit_information, fault_information)
```

But don't introduce three or four pieces of metadata merely because we previously proposed them.

The paper's requirement is:

> execution trace containing temporal coverage/fault feedback.

It does not prescribe a tuple structure. 

So the implementation should first establish a **reliable ordered edge trace**, then add additional event attributes only if they demonstrably improve the model.

---

# 7. Step 3 — Store the complete trace

For every seed, maintain something like:

```text
seed_id
trace
reward information
Q-value
```

Conceptually:

```text
SeedRecord
├── seed_id
├── trace
├── trace_length
├── q_value
└── metadata
```

For example:

```text
seed_0421
trace = [12, 18, 27, 31, 18, 45, ...]
trace_length = 18342
q_value = 0.73
```

The trace length is recorded because it is needed for variable-length RNN processing.

---

# 8. Step 4 — Convert edge IDs into tokens

The RNN should not interpret an integer edge ID as a continuous numerical value.

For example, don't do:

```text
edge 100 → 100.0
edge 101 → 101.0
edge 500 → 500.0
```

because that falsely implies that edge 500 is mathematically "larger" than edge 100.

Instead create a mapping:

```text
edge_id → token_id
```

For example:

```text
edge 12 → token 7
edge 18 → token 21
edge 27 → token 43
edge 31 → token 9
```

Then:

```text
execution trace

[12,18,27,31,18]
       ↓
tokenization
       ↓
[7,21,43,9,21]
```

---

# 9. Step 5 — Embedding layer

Now create a trainable embedding:

```text
token ID
   ↓
Embedding
   ↓
dense vector
```

Our current concrete implementation choice is:

\[
\boxed{\text{Embedding dimension}=32}
\]

So:

```text
token 7
   ↓
[0.13, -0.27, ..., 0.44]
       32 values
```

The embedding dimension **32 is our implementation parameter**, not something specified by the paper. The project context explicitly records this distinction. 

---

# 10. Step 6 — Feed the complete sequence to the GRU

Now the sequence becomes:

```text
token_1 → embedding → x1
token_2 → embedding → x2
token_3 → embedding → x3
...
token_N → embedding → xN
```

Then:

```text
x1 ──► GRU ──► h1
x2 ──► GRU ──► h2
x3 ──► GRU ──► h3
...
xN ──► GRU ──► hN
```

Our concrete implementation choice:

```text
single GRU layer
hidden size = 128
```

Therefore:

\[
h_N \in \mathbb{R}^{128}
\]

Again, **128 is our chosen GRU hidden size**, not the 128-D PPO coverage representation. That distinction is explicitly noted in the project context. 

---

# 11. Do NOT truncate the sequence

Under your current decision:

```text
Raw trace:
e1,e2,e3,...,e10000
```

becomes:

```text
RNN input:
e1,e2,e3,...,e10000
```

not:

```text
e1,...,e256
```

The GRU must process all events.

### But this creates an engineering problem

Very long traces mean:

- large computation;
- large activation memory during training;
- slower backpropagation;
- difficulty batching traces of dramatically different lengths.

That is a **performance problem**, not a reason to silently discard events.

So initially implement:

> **Full trace → sequential GRU processing.**

Then optimize the implementation if performance becomes unacceptable.

For batching, use variable-length sequence handling such as packed sequences rather than padding every seed to a globally fixed length.

---

# 12. Step 7 — Obtain the latent vector

After the final trace event:

\[
h_N
\]

is the seed's encoded execution representation.

For example:

```text
seed_0421
   │
   ▼
18,342 execution events
   │
   ▼
GRU
   │
   ▼
h_N =
[0.17, -0.32, 0.08, ..., 0.41]
       128 values
```

This is the point where the complete temporal execution history has been compressed into a fixed-dimensional representation.

This corresponds directly to:

> execution trace → RNN → encoded latent vector. 

---

# 13. Step 8 — Feed the latent vector to the DQN

Now:

```text
128-D latent vector
        │
        ▼
      DQN
```

There is **no PPO 64-D vector concatenated here**.

There is also **no campaign-state vector added at this stage** in our paper-faithful implementation.

A simple concrete DQN head can be:

```text
Linear(128 → 64)
      ↓
    ReLU
      ↓
Linear(64 → 1)
```

Therefore:

\[
h_N
\rightarrow
FC
\rightarrow
ReLU
\rightarrow
FC
\rightarrow
Q_i
\]

and:

\[
\boxed{Q_i=Q(s,\text{seed}_i)}
\]

The exact MLP dimensions are implementation choices because the paper does not specify the DQN architecture. 

---

# 14. Step 9 — Scalar output

For example:

```text
seed_0421 → Q = 0.73
```

Another seed:

```text
seed_0198 → Q = 0.42
```

Another:

```text
seed_0832 → Q = 1.15
```

The DL optimizer therefore creates:

```text
seed_0832 → 1.15
seed_0421 → 0.73
seed_0198 → 0.42
```

This is exactly the point of the module: the DQN predicts the expected long-term utility of mutating each seed, and AFL++ uses that predicted utility for prioritization. 

---

# 15. Step 10 — Use the Q-value for seed prioritization

The inference path is now complete.

```text
Seed A
  ↓
Trace
  ↓
RNN
  ↓
DQN
  ↓
Q(A)=0.73

Seed B
  ↓
Trace
  ↓
RNN
  ↓
DQN
  ↓
Q(B)=1.15

Seed C
  ↓
Trace
  ↓
RNN
  ↓
DQN
  ↓
Q(C)=0.42
```

Then:

```text
B → highest priority
A → second
C → third
```

---

# 16. Step 11 — Actually fuzz the selected seed

Suppose the optimizer selects:

```text
seed_B
```

AFL++ then mutates it.

This is where the PPO scheduler operates.

Conceptually:

```text
DL seed optimizer
        │
        ▼
"Use seed B"
        │
        ▼
PPO scheduler
        │
        ▼
"Use mutation strategy X"
        │
        ▼
AFL++ mutation
        │
        ▼
mutated input
```

So the two learners cooperate but have different responsibilities.

---

# 17. Step 12 — Execute the mutated seed

Suppose the mutation produces:

```text
seed_B_mut_17
```

AFL++ executes it.

We observe:

```text
old coverage = 4,210 edges
new coverage = 4,218 edges
```

and perhaps:

```text
crash = 0
```

Then:

\[
\Delta coverage=8
\]

---

# 18. Step 13 — Calculate the DL reward

We have agreed on:

\[
\boxed{
r =
w_{cov}\Delta coverage
+
w_{crash}I_{crash}
+
w_{novelty}I_{novel}
}
\]

The paper itself gives the same conceptual reward ingredients for the fuzzing framework—coverage, crash, and novelty signals. 

For the DL seed optimizer, the reward should correspond to the outcome of fuzzing the selected seed.

For example:

```text
Δcoverage = 8
crash = 0
novelty = 1
```

could produce:

```text
r = wcov*8 + wcrash*0 + wnovelty*1
```

The actual weights need to be fixed experimentally and consistently.

---

# 19. Step 14 — Create the DQN learning experience

This is where we need to be careful about the meaning of the DQN transition.

Conceptually:

```text
Before selecting seed:
        current situation
              │
              ▼
          select seed_i
              │
              ▼
          fuzz seed_i
              │
              ▼
            reward
              │
              ▼
      resulting situation
```

Store:

\[
(s_t,a_t,r_t,s_{t+1})
\]

where:

- \(s_t\) = current DQN decision context;
- \(a_t\) = selected seed;
- \(r_t\) = reward from fuzzing it;
- \(s_{t+1}\) = resulting context.

### Important implementation clarification

The paper's notation is somewhat underspecified because it describes the **seed execution trace as the RNN input**, while its Q-learning equation uses conventional \(Q(s,a)\) notation. It does not explicitly define a separate vector representation for `s`.

Therefore, **do not invent a second 64-D state just to satisfy the equation**.

For our implementation, maintain the seed's trace-derived latent representation as the neural representation used by the DQN and define the seed-selection transition at the **seed decision boundary**.

That keeps the implementation closest to Figure 2 and the textual description.

---

# 20. Step 15 — Replay buffer

The paper gives the Q-learning equation but does not specify experience replay.

For a neural DQN, we should add replay.

Use:

```text
ReplayBuffer capacity = 50,000
```

Store:

```text
(trace / encoded state,
 selected seed,
 reward,
 next decision representation)
```

The reason is that AFL++ produces highly correlated sequential experiences. Random replay batches reduce that correlation.

The project context explicitly identifies replay and a target network as necessary practical corrections to the paper's simple Q-learning rule. 

---

# 21. Step 16 — Wait until enough experiences exist

Don't train immediately from one transition.

For example:

```text
Replay buffer
     │
     ├── transition 1
     ├── transition 2
     ├── transition 3
     ├── ...
     └── transition N
```

Once you have enough samples:

```text
sample minibatch
```

For our implementation:

```text
batch_size = 32
```

This is an implementation choice, not a paper parameter.

---

# 22. Step 17 — Calculate current Q prediction

For each sampled experience:

```text
trace
 ↓
embedding
 ↓
GRU
 ↓
latent
 ↓
DQN
 ↓
Q_current
```

So:

\[
Q_{\theta}(s_t,a_t)
\]

is produced by the current online network.

---

# 23. Step 18 — Calculate the TD target

The paper gives:

\[
y =
r+\gamma\max_{a'}Q(s',a')
\]

with:

\[
\gamma=0.95
\]

and:

\[
\alpha=0.001
\]

as specified in Equation (3). 

For a neural implementation, use a frozen target network:

\[
\boxed{
y =
r+\gamma\max_{a'}Q_{\theta^-}(s',a')
}
\]

with:

```text
γ = 0.95
```

The target network is our implementation stabilization.

---

# 24. What does `max a'` mean here?

This is one of the tricky parts of implementing the paper.

The actions correspond to **candidate seeds**.

So conceptually:

```text
current queue:

seed A
seed B
seed C
seed D
...
```

For the next decision:

```text
Q(next_state, seed A)
Q(next_state, seed B)
Q(next_state, seed C)
Q(next_state, seed D)
...
```

and:

\[
\max_{a'}Q(s',a')
\]

means:

> choose the highest-valued candidate seed in the next decision.

Do not calculate this over an arbitrary mutation operator set—that is the PPO scheduler's action space.

---

# 25. Step 19 — Calculate loss

Use Huber loss:

\[
L = Huber(Q_{\theta}(s,a),y)
\]

The paper does **not specify Huber loss**; this is our neural-DQN stabilization choice.

The important conceptual objective is:

```text
predicted Q
     ↓
compare
     ↑
target Q
```

so that:

\[
Q_{\theta}(s,a)\rightarrow y
\]

---

# 26. Step 20 — Backpropagation

Now perform:

```text
loss
 ↓
backpropagation
 ↓
DQN weights
 ↓
GRU weights
 ↓
embedding weights
```

Therefore, **the entire trace encoder learns jointly with the DQN**.

This is important.

The embedding is not fixed.

The GRU is not fixed.

They are adjusted according to the seed-selection learning objective.

So eventually the system learns:

> Which temporal execution patterns are associated with seeds that produce high future reward?

---

# 27. Step 21 — Update the online network

Using Adam, update:

```text
Embedding parameters
GRU parameters
DQN parameters
```

We can use:

```text
learning_rate = 0.001
```

but be careful in your thesis:

The paper's \(\alpha=0.001\) is explicitly the learning rate in its Q-learning update equation. 

When translating that to a neural optimizer, we use it as the starting optimization learning rate, but this is technically an implementation mapping rather than evidence that the authors used Adam with exactly `lr=0.001`.

---

# 28. Step 22 — Update target network

Maintain:

```text
Online network
      │
      ├── Embedding
      ├── GRU
      └── DQN
```

and:

```text
Target network
      │
      ├── Embedding
      ├── GRU
      └── DQN
```

Periodically:

```text
target_network.load_state_dict(
    online_network.state_dict()
)
```

Our chosen synchronization interval:

```text
C = 1000 gradient updates
```

Again, this is **our stabilization mechanism**, not a parameter from the paper. 

---

# 29. The complete learning cycle

Now put everything together:

```text
                         AFL++ queue
                             │
                             ▼
                         Seed_i
                             │
                             ▼
                    Execute target
                             │
                             ▼
                  COMPLETE EXECUTION TRACE
                             │
                             ▼
                     Trace tokenization
                             │
                             ▼
                      Edge embeddings
                             │
                             ▼
                          GRU
                             │
                             ▼
                     Latent vector
                             │
                             ▼
                           DQN
                             │
                             ▼
                         Q(seed_i)
                             │
                             ▼
                    Seed prioritization
                             │
                             ▼
                      Select seed_i
                             │
                             ▼
                       PPO scheduler
                             │
                             ▼
                    Select mutation op
                             │
                             ▼
                          Mutate
                             │
                             ▼
                    Execute mutation
                             │
                             ▼
                  coverage/crash/novelty
                             │
                             ▼
                          Reward
                             │
                             ▼
                   Create transition
                             │
                             ▼
                     Replay buffer
                             │
                             ▼
                      Sample batch
                             │
                             ▼
                  Online Q prediction
                             │
                             ▼
                   Target Q calculation
                             │
                             ▼
                         Huber loss
                             │
                             ▼
                      Backpropagation
                             │
                             ▼
               Embedding + GRU + DQN update
                             │
                             ▼
                    Target sync periodically
                             │
                             └──────────► next cycle
```

---

# 30. Where each piece should physically live

I recommend dividing the implementation into four components.

### A. AFL++ / instrumentation layer

Responsible for:

```text
seed execution
    ↓
ordered trace collection
    ↓
coverage
    ↓
fault/crash information
```

The trace collector belongs here/alongside the instrumentation mechanism. 

---

### B. DL optimizer server

Responsible for:

```text
trace reception
     ↓
tokenization
     ↓
embedding
     ↓
GRU
     ↓
DQN
     ↓
Q-value
```

and:

```text
replay buffer
training
target network
model checkpoint
```

---

### C. AFL++ integration interface

The paper describes the use of an external mutator interface and UNIX-domain sockets for communication between AFL++, learning agents and the mutator. 

Conceptually:

```text
AFL++ C/C++
     │
     │ trace / seed query
     ▼
UNIX socket
     │
     ▼
Python DL optimizer
```

---

### D. PPO module

Keep this separate:

```text
coverage 128-D
+
mutation history 64-D
        ↓
     PPO
        ↓
mutation action
```

Do not feed this state into the DL optimizer.

---

# 31. Recommended message flow

A practical communication protocol can look like:

### Seed evaluation

```text
AFL++ → DL:

SEED_TRACE
seed_id
trace_length
trace_events
```

DL responds:

```text
DL → AFL++:

SEED_SCORE
seed_id
q_value
```

---

### Training feedback

After the seed is fuzzed:

```text
AFL++ → DL:

SEED_RESULT
seed_id
reward
new_coverage
crash
novelty
next_seed_context
```

The DL process then:

```text
store transition
 ↓
replay buffer
 ↓
training when ready
```

---

# 32. What happens when a new AFL++ seed is discovered?

Suppose mutation produces:

```text
new_seed_9001
```

because it found novel coverage.

AFL++ executes it and records:

```text
trace(new_seed_9001)
```

Then:

```text
new_seed_9001
       ↓
complete trace
       ↓
GRU
       ↓
latent
       ↓
DQN
       ↓
Q(new_seed_9001)
```

Cache:

```text
seed_9001 → Q-value
```

so AFL++ does not need to recompute it unnecessarily.

---

# 33. What happens when an old seed is revisited?

Do **not** assume that its Q-value remains permanently valid.

The network changes during training.

Therefore:

```text
old cached Q
```

is an inference result from an older version of the model.

You need a cache policy such as:

```text
seed_id
model_version
q_value
```

For example:

```text
seed_0421
model_version = 37
Q = 0.73
```

When the model reaches version 38, decide whether to recompute scores.

This is better than simply caching Q forever.

---

# 34. The most important implementation files/modules

A clean project structure could be:

```text
dl_seed_optimizer/
│
├── trace/
│   ├── collector
│   ├── serializer
│   └── tokenizer
│
├── model/
│   ├── embedding.py
│   ├── gru_encoder.py
│   ├── dqn.py
│   └── seed_q_network.py
│
├── training/
│   ├── replay_buffer.py
│   ├── target_network.py
│   ├── loss.py
│   └── trainer.py
│
├── inference/
│   ├── scorer.py
│   └── ranking.py
│
├── communication/
│   └── unix_socket.py
│
└── checkpoints/
```

This is not mandated by the paper; it is simply a clean implementation organization.

---

# 35. Implementation order I recommend

Do **not** implement everything simultaneously.

Follow this order.

### Phase 1 — Trace collection

First make this work:

```text
seed
 ↓
target execution
 ↓
ordered trace
 ↓
print/save trace
```

Before touching PyTorch.

Verify that:

- events are actually ordered;
- each seed gets a distinct trace;
- repeated execution gives reasonably consistent traces;
- trace length is recorded;
- crash executions don't corrupt the trace;
- the trace is available outside the target process.

---

### Phase 2 — Trace → tokens

Implement:

```text
raw trace
 ↓
edge IDs
 ↓
token IDs
```

Verify:

```text
seed A:
[12,18,27,31,...]

→

[7,21,43,9,...]
```

---

### Phase 3 — RNN only

Build:

```text
Embedding(32)
      ↓
GRU(128)
      ↓
latent(128)
```

Feed a few real traces.

Verify:

```text
trace
 ↓
latent
```

with no NaNs and correct dimensions.

---

### Phase 4 — DQN inference

Add:

```text
latent(128)
      ↓
Linear
      ↓
ReLU
      ↓
Linear
      ↓
scalar Q
```

Verify:

```text
trace A → Q_A
trace B → Q_B
trace C → Q_C
```

---

### Phase 5 — AFL++ ranking

Connect:

```text
AFL++ → trace → DL → Q
```

and test whether AFL++ can obtain:

```text
seed_id → Q-value
```

without training.

At this stage you have a **working inference-based seed prioritizer**.

---

### Phase 6 — Reward collection

Only after inference works:

```text
selected seed
 ↓
mutation
 ↓
execution
 ↓
coverage/crash/novelty
 ↓
reward
```

Implement and verify reward values independently.

---

### Phase 7 — Replay buffer

Add:

```text
transition
 ↓
replay buffer
```

Check that transitions are correctly stored and retrieved.

---

### Phase 8 — DQN training

Add:

```text
batch
 ↓
Q_current
 ↓
Q_target
 ↓
Huber loss
 ↓
backprop
 ↓
Adam
```

Initially test training offline with stored traces/transitions before putting training directly into the live fuzzing loop.

---

### Phase 9 — Target network

Add:

```text
online Q
target Q
```

and periodic synchronization.

---

### Phase 10 — Asynchronous live training

Finally:

```text
AFL++ process
       │
       │ UNIX socket
       ▼
DL inference worker
       │
       ▼
training worker
```

so PyTorch training doesn't block the AFL++ execution loop.

The paper itself emphasizes the external interface/UNIX-socket architecture and asynchronous/modular deployment. 

---

# 36. Final parameter table

I would currently lock the following:

| Component | Final implementation |
|---|---|
| Seed representation | Complete ordered execution trace |
| Trace truncation | **None** |
| Trace ordering | Preserve execution order |
| Trace collector | AFL++ instrumentation/execution layer |
| Trace event | Start with edge ID |
| Tokenization | Edge ID → token ID |
| Embedding | 32-D |
| RNN | Single GRU |
| GRU hidden size | 128 |
| Latent vector | 128-D |
| Campaign-state vector | **None** |
| PPO 64-D history | **Not used** |
| DQN input | GRU latent |
| DQN output | One scalar Q |
| DQN hidden layer | 64-D ReLU baseline |
| Learning rate | 0.001 starting value |
| Discount | 0.95 |
| Replay buffer | 50,000 |
| Batch size | 32 |
| Target network | Yes |
| Target synchronization | 1,000 gradient updates |
| Loss | Huber |
| Seed ranking | Descending Q |
| Communication | UNIX-domain socket |
| Training | Asynchronous |

The paper explicitly supports the execution-trace → RNN → latent → DQN → seed-ranking architecture and the \(\alpha=0.001,\gamma=0.95\) Q-learning formulation.  The remaining neural architecture and stabilization values are implementation decisions and should be described as such. 

---

## The one thing I would **not** implement yet

I would **not yet code the 256-event truncation, 64-D campaign state, or PPO-history concatenation**.

Those were earlier implementation proposals, and based on the paper plus our subsequent decisions, they are unnecessary.

The clean implementation target is:

\[
\boxed{
\text{Complete ordered trace}
\rightarrow
\text{Embedding}
\rightarrow
\text{GRU}
\rightarrow
\text{latent}
\rightarrow
\text{DQN}
\rightarrow
\boxed{Q(\text{seed})}
}
\]

Then the learning loop is:

\[
\boxed{
Q(\text{seed})
\rightarrow
\text{select seed}
\rightarrow
\text{mutate}
\rightarrow
\text{execute}
\rightarrow
r
\rightarrow
\text{replay}
\rightarrow
\text{TD target}
\rightarrow
\text{loss}
\rightarrow
\text{backpropagation}
\rightarrow
\text{update Embedding+GRU+DQN}
}
\]

That is the implementation path I would follow from here.

# Create actions.md

record what actions, decision has been taken throughout the immplmentation and create actions.md file consisting all those records. Include the tools installed, used, command executed, hyperparameter values used, algorithms used like losses, optimizers, and the training and inference flow.
