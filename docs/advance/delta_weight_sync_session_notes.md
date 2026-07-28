# Delta Weight Sync Experiment and Pipeline Notes

Last updated: 07/27/2026.

This document summarizes an extended debugging and design discussion around the
`delta-weight-sync` branch. It is not a verbatim transcript. It records the
experiment setup, implementation details, profiling caveats, correctness checks,
and the NCCL bucket pipeline conclusions reached during the session.

## 1. Experiment setup

The initial experiment used:

- Model: Qwen3-0.6B.
- Algorithm: GRPO on GSM8K.
- Trainer: `OneStepOffRayTrainer`.
- Actor: FSDP2.
- Rollout: SGLang.
- Deployment: disaggregated trainer and rollout with one-step-off policy.
- Weight-sync backends compared:
  - `nccl`
  - `delta`
  - `delta_sharded`

The trainer entry point is:

```text
verl.experimental.one_step_off_policy.main_ppo
```

It constructs:

```text
OneStepOffRayTrainer
  └── SeparateRayPPOTrainer
```

This is the experimental separated trainer, not the standard
`verl.trainer.ppo.ray_trainer.RayPPOTrainer`.

The main two-node recipe is:

```text
verl/experimental/one_step_off_policy/shell/
  grpo_0.6b_gsm8k_fsdp2_sglang_delta_2_6.sh
```

The experiment evolved from a 2-trainer-GPU + 6-rollout-GPU recipe to a
two-node setup with one full trainer node and one full rollout node. A separate
single-node 4-trainer-GPU + 4-rollout-GPU debug recipe was also created.

## 2. Weight-sync backends

### 2.1 Full NCCL

The `nccl` backend exports the current full actor weights, assembles them into
fixed-size buffers, and broadcasts every buffer to rollout workers.

Conceptually:

```text
FSDP shard
  → full_tensor() all-gather
  → fill send bucket
  → NCCL broadcast
  → rollout receive buffer
  → SGLang load_weights
```

### 2.2 Base delta

The base `delta` backend keeps the previous synchronized full weights in pinned
CPU memory. For every later synchronization it:

```text
current FSDP shards
  → full_tensor() all-gather
  → previous snapshot H2D
  → bytewise diff on GPU
  → encode changed positions and values
  → NCCL broadcast sparse payload
  → decode and masked apply in SGLang
```

Diffing is bit-exact. There is no floating-point threshold: each element is
viewed as an integer of the same width and compared for exact inequality.

### 2.3 Sharded delta

`delta_sharded` moves diffing below the full-parameter all-gather:

```text
each FSDP rank:
  local current shard
  + local pinned-CPU snapshot
  → local bytewise diff
  → gather only changed positions and values to rank 0
  → broadcast the same sparse wire format
```

Compared with base delta:

- Snapshot memory is sharded across FSDP ranks.
- Full unchanged parameters are not gathered.
- Gather traffic scales with the changed ratio instead of full model size.
- Receiver wire format and SGLang apply path remain unchanged.

## 3. Delta snapshot and FSDP offload

The base delta implementation deliberately stores the previous synchronized
weights in CPU pinned memory. On each synchronization, snapshot chunks are
prefetched back to GPU for diffing.

Without actor parameter offload:

```text
previous snapshot: CPU → GPU
current parameter: already on GPU
→ GPU diff
```

With actor parameter offload:

```text
previous snapshot: CPU → GPU
current FSDP shard: CPU → GPU
→ optional full_tensor() all-gather
→ GPU diff
```

Therefore `param_offload=True` adds H2D traffic for the current parameters.
That traffic may contend with snapshot H2D and increases the weight-sync
critical path.

For `delta_sharded`, offload still requires staging each local current shard
from CPU to GPU, but it does not require gathering a full parameter first.

`optimizer_offload` is different. Optimizer state is not sent to rollout, so it
normally affects weight-sync time only indirectly through phase transitions,
memory pressure, or the surrounding timing scope.

The primary experiment disabled actor parameter and optimizer offload. Reference
policy offload does not directly affect actor-to-rollout weight synchronization.

## 4. Delta chunk pipeline

The main implementation is split across:

```text
delta_sync/wrapper.py
  Scheduling of prefetch, diff, snapshot update, encode, and buckets.

delta_sync/delta_state.py
  Pinned snapshots, H2D/D2H streams, and CUDA events.

delta_sync/encode.py
  Bytewise diff encoding and bucket assembly.
```

The implementation uses:

- A dedicated H2D stream for old snapshot prefetch.
- The current/default stream for diff and encode.
- A dedicated D2H stream for updating the CPU snapshot.
- CUDA events to express stream dependencies.

A simplified steady-state schedule is:

```text
H2D snapshot(N+1)
       overlaps
diff(N) → enqueue D2H snapshot update(N) → encode(N)
```

`compute_diffs(N)` depends on the previously submitted H2D for snapshot chunk
`N`. It does not depend on H2D for chunk `N+1`.

The current Python loop has one-step lookahead. It submits the next snapshot
prefetch before processing the pending chunk. Moving to a deeper lookahead could
submit later H2D operations earlier, but that does not guarantee a speedup:

- The H2D stream itself is serial.
- H2D and diff/encode may contend for HBM bandwidth.
- A deeper lookahead keeps more GPU tensors alive.
- Larger outstanding chunks increase peak GPU memory.

## 5. Why process delta in chunks?

Processing the entire model at once would require simultaneously retaining:

```text
current full weights
+ all previous GPU snapshot copies
+ all diff masks
+ all encoded outputs
```

Chunking bounds memory and enables pipelining:

```text
H2D(N+1) overlaps diff/encode(N)
D2H(N) overlaps later compute
encoded chunks can enter a send bucket early
```

The trade-off is:

```text
smaller chunks:
  lower memory, finer overlap, more Python/kernel-launch overhead

larger chunks:
  better per-operation efficiency, coarser overlap, higher memory
```

Increasing `chunk_params` groups more parameters into one encoder call, but the
current diff path still launches a comparison per tensor. It does not
automatically become one fused multi-tensor CUDA kernel.

## 6. HBM, PCIe, NVLink, and overlap

HBM is the high-bandwidth memory physically attached to training GPUs. “HBM
bandwidth” is the rate at which GPU engines can read and write that memory.

The H2D path is:

```text
CPU memory → PCIe/NVLink → GPU HBM
```

H2D uses:

- The host-to-device interconnect.
- A GPU copy engine.
- Some HBM write bandwidth.

The bytewise diff reads the current parameter and old snapshot from HBM and
writes a boolean mask, so it is often HBM-bandwidth intensive.

Theoretical peak bandwidth is only an upper bound. A task may use less than the
peak because it has too few parallel requests, poor access patterns,
dependencies, compute between accesses, or another bottleneck such as PCIe.

If two tasks individually saturate the same resource, overlap does not reduce
their combined data-transfer time:

```text
total time = total bytes / saturated bandwidth
```

If they use different bottlenecks or leave capacity unused, overlap can reduce
wall-clock time. H2D is frequently limited by PCIe or NVLink while diff is
limited by HBM, so they may overlap profitably even though both touch HBM.

## 7. Profiling

The experiment branch supports:

```text
VERL_SYNC_PROFILE=1
VERL_PROFILE_NCCL_SEND=1
VERL_PROFILE_DELTA_SEND=1
```

The script can propagate all three through the Ray runtime environment. The
active backend uses its relevant probes and ignores the others.

Typical NCCL sender output is:

```text
[nccl-send-profile]
v=...
buckets=...
pull(gather)=...
fill=...
sync=...
wait(broadcast)=...
total=...
```

Important interpretation:

- `pull(gather)` measures Python time driving a lazy generator. CUDA collectives
  may only be submitted during this interval.
- `fill` measures submission of copies into the send bucket.
- `sync` waits for queued CUDA work and may include gather, fill, and an
  overlapping previous broadcast.
- `wait(broadcast)` is only the broadcast wait not already hidden elsewhere.

Consequently:

```text
pull(gather) is not necessarily the complete GPU all-gather time
wait(broadcast) is not necessarily the complete broadcast time
```

Profiling adds synchronization and can change overlap. Backend comparisons
should primarily use unprofiled end-to-end:

```text
timing_s/sync_rollout_weights
```

Detailed attribution should use CUDA events on the actual gather and NCCL
streams, or an Nsight Systems timeline.

### CUDA events

A CUDA event is a marker recorded in a CUDA stream. With timing enabled, two
events measure GPU execution time rather than Python submission time:

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record(stream)
do_gpu_work()
end.record(stream)

end.synchronize()
milliseconds = start.elapsed_time(end)
```

Events can also express dependencies without synchronizing the entire device:

```python
ready.record(producer_stream)
consumer_stream.wait_event(ready)
```

## 8. Delta correctness and checksums

The receiver automatically verifies a checksum for every delta flush. The
sender hashes:

```text
positions + changed values
```

The SGLang custom loader hashes the received payload and raises on mismatch.

“Zero receiver checksum failures across all 400 delta syncs” means that every
observed receiver checksum matched its sender checksum. It provides evidence
that the encoded payload was not corrupted in transport.

It does not by itself prove that final rollout weights equal full NCCL weights.
A matching checksum cannot detect:

- Incorrect positions produced by the sender.
- A wrong parameter name or manifest offset.
- A decode/apply bug.
- A mismatched initial rollout model.
- Correct bytes applied to the wrong live parameter.

The branch contains bit-identity tests for:

- Delta encode/decode round trips.
- SGLang masked apply.
- Dense first-sync apply.
- Sharded-delta positions and values.
- Multi-GPU sharded diff against a full-gather baseline.

There is no production checker that reads every live SGLang parameter after
every synchronization and compares it bitwise with trainer parameters.

A strict debug checker would periodically compute corresponding per-parameter
hashes on:

```text
trainer current weights
rollout live SGLang weights after apply
```

This is expensive and complicated by tensor parallelism and SGLang’s internal
weight layouts, so it should be a debug-only feature.

## 9. Full NCCL bucket pipeline

The NCCL checkpoint engine uses two fixed GPU buffers and an asynchronous
`BroadcastOperation`.

At a high level:

```text
broadcast bucket N
       overlaps
produce bucket N+1:
  FSDP full_tensor() → split chunks → fill send buffer
```

The sender:

1. Pulls weight chunks from a lazy generator.
2. Fills the current send buffer.
3. Synchronizes at a bucket boundary.
4. Waits for the prior broadcast task.
5. Starts broadcasting the filled buffer.
6. Swaps to the second buffer.
7. Continues producing the next bucket.

The receiver:

1. Receives one bucket.
2. Swaps buffers.
3. Immediately posts receive for the next bucket.
4. Yields the previous bucket to SGLang.
5. Waits before reusing the old buffer.

This creates two overlaps:

```text
trainer:
broadcast(N) overlaps gather/fill(N+1)

rollout:
receive(N+1) overlaps load/apply(N)
```

The implementation does not intentionally run multiple sender broadcast
collectives concurrently.

## 10. Submission versus completion

Calling:

```python
collective.broadcast(bucket)
```

normally submits NCCL work to a CUDA stream. The Python call may return before
the GPU and network operation finishes.

The relevant milestones are:

```text
1. Trainer submits broadcast(N).
2. Rollout ranks submit the matching broadcast(N).
3. NCCL kernels execute and move data.
4. Trainer-local NCCL operation completes.
5. Rollout-local receive operations complete.
6. SGLang finishes loading the bucket.
```

All ranks in a communicator must invoke collectives in the same order. NCCL
matches calls by communicator and sequence, not by an application-level bucket
name:

```text
first broadcast  ↔ first broadcast on every rank
second broadcast ↔ second broadcast on every rank
```

If the trainer submits a later collective before a rollout rank submits its
matching call, the NCCL kernel cannot complete. The Python API may have returned
after enqueueing, but a later CUDA synchronization will block until the
collective can make progress.

“Trainer-local broadcast completion” means:

```text
the trainer NCCL stream has completed its local operation
and the source buffer can be safely reused
```

It is stronger than “both sides called the API,” but it is not an
application-level acknowledgment that SGLang has loaded the weights.

In the sender loop, `torch.cuda.synchronize()` is unconditional at bucket
boundaries, even when profiling is disabled. Profiling adds another
synchronization inside `BroadcastOperation`.

## 11. Double-buffer backpressure

Only two rollout receive buffers are available:

```text
Buffer A: bucket N being consumed by SGLang
Buffer B: bucket N+1 being received
```

This bounds how far the network can run ahead of model loading.

If SGLang load is slower than broadcast:

```text
both buffers remain occupied
→ rollout cannot post receive for bucket N+2
→ trainer's matching collective cannot complete
→ backpressure reaches the sender
```

The steady-state time per bucket is therefore approximately:

```text
max(
  gather + fill,
  broadcast + receive,
  SGLang load
)
```

When SGLang load is shorter than broadcast, loading can be hidden behind receive.
When it is longer, rollout apply becomes the bottleneck.

## 12. What “pull the next chunk before closing a bucket” means

The sender checks overflow only after requesting the next chunk:

```text
pull next chunk
→ check current_offset + chunk_size
→ if it does not fit, broadcast the current bucket
→ swap buffers
→ put the already-pulled chunk into the next bucket
```

For example, with a 100 MB bucket:

```text
current bucket:
┌──────────────┬──────────────┬─────────┐
│ A: 40 MB     │ B: 40 MB     │ free 20 │
└──────────────┴──────────────┴─────────┘

pull C: 30 MB

80 MB + 30 MB > 100 MB
```

The engine seals the current bucket containing A and B, starts its broadcast,
switches buffers, and places C into the next bucket.

Pulling C may already trigger C’s `full_tensor()` all-gather if C belongs to a
new parameter. Subsequent parameters needed to fill the next bucket are pulled
while the previous bucket broadcasts.

The producer is lazy:

```text
request one parameter
→ full_tensor() all-gather for that parameter
→ expose its byte chunks one at a time
→ request the next parameter only after those chunks are consumed
```

It does not pre-gather multiple future buckets. A single parameter larger than
the bucket size is a special case: one full tensor can be sliced across multiple
buckets.

“Close” or “seal” a bucket means:

- Stop adding chunks to it.
- Associate its metadata with the broadcast.
- Hand its buffer to `BroadcastOperation`.
- Swap to the other GPU buffer.
- Reset offset and metadata for the next bucket.

## 13. Four-bucket example when broadcast is slower

Let:

```text
G = gather + fill one bucket = 1 second
B = broadcast + receive one bucket = 3 seconds
```

Assume SGLang load is no slower than broadcast.

The idealized schedule is:

```text
t=0..1
  gather/fill bucket 1

t=1..4
  broadcast/receive bucket 1
  gather/fill bucket 2 during t=1..2
  wait for bucket 1 during t=2..4

t=4..7
  broadcast/receive bucket 2
  gather/fill bucket 3 during t=4..5
  wait for bucket 2 during t=5..7

t=7..10
  broadcast/receive bucket 3
  gather/fill bucket 4 during t=7..8
  wait for bucket 3 during t=8..10

t=10..13
  broadcast/receive bucket 4
```

Timeline:

```text
time:        0   1   2   3   4   5   6   7   8   9  10  11  12  13

gather/fill: [B1]
                 [B2]        [B3]        [B4]

broadcast:       [------B1------]
                                 [------B2------]
                                                 [------B3------]
                                                                 [------B4------]
```

The pipeline takes approximately:

```text
G + 4B = 13 seconds
```

Fully serial execution would take:

```text
4G + 4B = 16 seconds
```

Because `B > G`, the producer repeatedly finishes early and waits at the next
bucket boundary. Broadcast is the bottleneck in this example.

## 14. Which is usually slower: broadcast or all-gather + fill?

There is no topology-independent answer.

For an ideal large-message collective with the same effective link bandwidth:

```text
AllGather:
t ≈ S/BW × (N-1)/N

Broadcast:
t ≈ S/BW
```

For 16 ranks, ideal broadcast is only about `16/15`, or 6.7%, slower than ideal
all-gather. Broadcast has a root injection bottleneck, while all-gather input is
distributed across ranks.

Factors that can make broadcast slower:

- One root must originate the full bucket.
- The rollout group may span more nodes.
- Root GPU/NIC bandwidth may be the bottleneck.

Factors that can make FSDP all-gather + fill slower:

- FSDP exports are per parameter rather than one contiguous bucket collective.
- Smaller collectives incur more launch latency.
- FSDP wrapping controls gather granularity.
- Gathered tensors must be copied into the send buffer.
- Concurrent gather and broadcast may contend for trainer NIC/GPU resources.

The correct method is to benchmark the actual placement and inspect a real GPU
timeline, not infer the bottleneck from model size alone.

## 15. Existing experiment observations

One nine-run comparison used steady-state steps 5–12:

```text
Qwen2.5-32B, 2 trainer + 2 rollout nodes
  NCCL:          7.632 s average
  Delta:         7.650 s average
  Delta Sharded: 2.964 s average

Qwen2.5-7B, 1 trainer + 1 rollout node
  NCCL:          1.517 s average
  Delta:         1.759 s average
  Delta Sharded: 1.035 s average

Qwen3-0.6B
  Delta Sharded was also the fastest backend.
```

The 32B recipe used:

```text
update_weights_bucket_megabytes=512
```

Its NCCL run had profiling disabled, so the results do not contain clean
per-bucket all-gather and broadcast durations. The total 7.632 seconds cannot
establish whether the NCCL run was gather-bound or broadcast-bound.

The similar 32B totals for full NCCL and base delta also do not isolate
all-gather time because base delta adds snapshot H2D, diff, encode, sparse
broadcast, decode, and apply.

## 16. Practical conclusions

1. Use unprofiled end-to-end synchronization time for backend comparisons.
2. Use CUDA events or Nsight Systems for stage attribution.
3. Do not interpret Python submission time as GPU operation time.
4. Keep actor parameter offload fixed across backend experiments.
5. Compare identical model, topology, bucket size, and training steps.
6. Treat the receiver payload checksum as a transport-integrity check, not a
   complete end-to-end model equality proof.
7. The current NCCL design pipelines one bucket’s broadcast with production of
   the next bucket and pipelines rollout receive with previous-bucket apply.
8. The slowest of producer, network, and SGLang apply determines steady-state
   throughput.
