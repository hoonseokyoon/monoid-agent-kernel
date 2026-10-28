# Streaming and Stop

Choose who owns execution before connecting streamed output to a transport. A request can own
the run and join its cleanup when it ends. A background run can outlive individual readers.
Both paths have offline examples using real kernel and Reference components.

## Install and choose an owner

Use Python 3.11+ and install this checkout with `python -m pip install .`, or use a wheel from
the same revision. Run the repository examples from the checkout root. These examples use the
0.25.0 API; the [release audit](../V0_25_RELEASE_AUDIT.md) records artifacts and publication status.

| Product interaction | Execution owner | Disconnect and Stop |
|---|---|---|
| Request-scoped response | Host's `async with loop.astream(...)` | Early exit requests cancellation and joins stream cleanup |
| Background work with reconnecting readers | Host's submitted backend run | Reader close releases its subscription; authenticated `cancel_run(run_id, token)` explicitly cancels work |

The examples use temporary workspace/run roots and scripted adapters. Replace those with product
storage and a model adapter at `AgentLoop.from_config` or the backend's model factory. See the
[custom adapter example](../../examples/custom_model_adapter.py) and [gateway walkthrough](../BACKEND.md).
Your HTTP/WebSocket/SSE handler supplies its disconnect signal and owns delivery to the client.

## Stop a request stream

Run the [request example](../../examples/termination_budget.py):

```bash
python examples/termination_budget.py
python examples/termination_budget.py --scenario silent_stop
python examples/termination_budget.py --scenario consumer_cancel
```

The default leaves the stream after `Working`. Its output shows `cooperative_wait` during exit
and `settled` afterward, one receipt, `requested_by="user_cancel"`, and
`ended_by="provider_terminal"`. Reported usage is input 0/output 3; total-token coverage is
`unknown`. Later provider text stays undelivered. Silent Stop exercises cancellation before any
text arrives. Consumer cancellation reaches the caller as `CancelledError` after cleanup, and
the CLI supervisor displays it.

The example's `run_request` owns the loop, stream, progress task, control task, and model-I/O
subscription. Its `finally` releases provider gates, joins helper tasks, and closes the loop.
An unsuccessful model turn raises `TurnNotSettled(stream.suspension)`. Wire the same ownership
boundary to your transport, including failure before the first delta.

## Show Stop progress

Keep the `RunStream` handle and poll it from a separate progress task while the context exits.
The read-only `termination_budget()` API also exists on `AgentLoop` and `ModelCallRunner`.
This excerpt observes an existing stream:

```python
snapshot = stream.termination_budget()
for call in snapshot.calls:
    print(call.call_id, call.phase, call.deadline_remaining_s, call.drain_remaining_s)
    for window in call.grace_windows:
        print(window.source, window.budget_s, window.remaining_s)
exit_wait = snapshot.stream_exit
print(exit_wait.phase, exit_wait.hard_cancel_remaining_s)
```

Queries return frozen process-local snapshots and invoke no cancellation or host configuration
getter. A finite grace budget with `remaining_s=None` is reserved; `None`/`None` means unbounded.
Closing can coexist with a live drain window. Empty `calls` describes the owner's model calls;
tools, child runs, and resource cleanup have separate owners.

| Exit phase | Interpretation |
|---|---|
| `idle` | Exit wait has not started |
| `cooperative_wait` | Waiting for cooperative completion; the countdown can reach zero before cancellation is requested |
| `wait_interrupted` | The exit waiter was cancelled; its timer stopped, and direct re-entry reuses the original cutoff |
| `hard_cancel_requested` | Driver cancellation was requested; cleanup and settlement can continue |
| `settled` | The driver has settled; a zero remainder preserves a prior hard-cancel request, while `None` records completion without it |

Use the phase to observe a cancellation request. A zero countdown alone does not establish that
request or complete resource cleanup. Read available call usage from the final receipt.

## Collect usage during a bounded drain

A Stop can close the provider stream before its usage report arrives. Opt into a drain when the
host wants time to collect that report. These settings apply to an existing loop or direct runner:

```python
loop.async_model_abort_drain_s = 5.0  # read live; default 0 closes at Stop
runner = ModelCallRunner(adapter=adapter, abort_drain_s=5.0)
```

`interrupt_turn()`/`should_abort` and token `user_cancel` can start a drain. The same adapter stream
continues toward its end without delivering further content. The window closes at the earlier
of Stop plus the selected budget or the run deadline; the call keeps its Stop outcome.
`graceful_drain`, `deadline`, `host_shutdown`, and lease loss stop immediately unless `user_cancel`
already started a drain. In that case lease loss cuts it short, and the run deadline still bounds it.
The token preserves its first cause.

An early `astream` exit freezes its wait from currently owned calls: 8 seconds plus the largest
remaining drain-and-cleanup allowance. An active one-shot call contributes cleanup reservation;
an empty call set contributes no extra allowance. Consumed drain time and future configured drain
add no time. Later setting changes leave the first exit cutoff fixed; phase-specific cleanup
continues to sample its own grace. After the cutoff, the stream requests driver cancellation and
awaits cleanup and settlement.

Drain settings cover the loop's own model calls. Child/fork loops retain their default drain
budget of zero. `interrupt_turn()` does not reach a child; its streamed call continues, and
reported usage counts. Adapter-internal retries inside `astream_turn` can continue during drain;
the adapter owns stopping those retries and any resulting provider charges.

## Read final facts and usage

The request example receives settled receipts through `ModelIOSubscription`. After cleanup it
reads the single receipt's `termination` and `usage_evidence`. In product code, handle unavailable
facts explicitly: old records and adapters without observation metadata can lack them.

`termination` records logical `outcome`, first accepted stop `requested_by`, and provider-read
`ended_by` independently. `usage_evidence.reported` contains known counter sums; `completeness`
is `no_dispatch`, `complete`, `partial`, or `unknown` for each counter. Explicit zero is a report.
Kernel dispatches define the evidence boundary; provider SDK retries remain outside that count.

Keep numeric usage as the accounting input. Loop results, `metrics.updated`, checkpoint totals,
and token budgets include recorded drained usage. Direct runner callers can use
`provider_usage_of(exc)` on escaping `ModelCallAborted` or `RunCancelled`; failed receipts carry
the same numeric usage. `ValidatedCallRunner` keeps earlier receipts on `exc.receipts`.
Empty usage means no report was available before reading ended. Provider reports describe observed
usage; the host reconciles its bill separately.

Run [model_call_facts.py](../../examples/model_call_facts.py) for sparse observations and
[recover_model_call_facts.py](../../examples/recover_model_call_facts.py) for an in-memory lifecycle
round trip. The [model adapter contract](../CONTRACTS.md#model-adapter) defines the full facts and
accounting rules, and [Observability](../OBSERVABILITY.md) covers durable receipt projections.

## Observe a background run

Run the [background example](../../examples/observe_background_run.py):

```bash
python examples/observe_background_run.py
```

It submits a streaming run to the Reference backend with run-scoped credentials. Closing one
reader leaves its peer receiving the next delta. Reconnecting with `after_cursor=reader.cursor`
replays the retained suffix. The example overflows a two-frame ring, receives `cursor_gap`,
and marks `history_complete=false`. It then calls authenticated `cancel_run`, waits for terminal
state and public result readiness, and closes every reader.

Expected fields include `reader_closed=true`, `continued_after_disconnect=true`,
`reset_reason="cursor_gap"`, `cancel_requested=true`, and `state="cancelled"`.
Its failure cleanup releases provider gates, drains the backend, and waits for result publication
before temporary state is removed.

The live broker is a bounded process-local ring. Handle generation changes and cursor gaps in
the product. Show incomplete history or rebuild the display from separately enabled, authorized
durable content. For process replacement, use the handbook's
[durable private model streams](../EMBEDDING.md#durable-private-model-streams).

## Resume event delivery

Live content uses a generation/sequence cursor. Durable event delivery uses integer event sequences
through `EventSubscription` and `SequenceCursor`. The latter stores the next required sequence,
suppresses replay, and raises on a gap. With an existing authenticated Reference backend:

```python
subscription = backend.subscribe_events(run_id, run_token, from_seq=1)
page = subscription.poll(limit=500)
next_seq = page["next_seq"]
```

For SSE, send event sequence as `id`, accept `Last-Event-ID`, and resume at the next sequence.
Heartbeats have no event ID. A terminal subscription drains the event log once more before its
end frame. Persist the cursor when delivery must survive client restarts. The
[hosted example](../../examples/embedding_hosted_product.py) verifies duplicate-free polling and
fresh-backend reads of durable history with the original authorized token.

## Deadline inputs and recovery

Standalone `ModelCallRunner.acall` and `ValidatedCallRunner.acall` accept Unix epoch deadlines,
for example `deadline=time.time() + 30`. The runner converts once to a monotonic budget covering
retries and validated repairs. `None`/`math.inf` is unbounded; expired epochs and `-math.inf`
expire immediately. NaN, booleans, nonnumeric values, and overflowing integers raise `ValueError`
before adapter entry or receipt callbacks.

`RunLimits.max_duration_s` measures active time after workspace/resource setup. Model waits,
tool waits, drain, and duration metrics share that monotonic clock. Checkpoints store remaining
duration and restore it against the new process clock; parked time leaves the budget unchanged.
Settled-call recovery publishes authoritative evidence before an expired run boundary is applied.
See [recovery and upgrades](../EMBEDDING.md#checkpoints-recovery-and-upgrades) for durable-host obligations.
