# JumpProcessor — native ring + background-thread full model

## Why

The Wallaby jump-detection pipeline runs two models per cycle:
- **backbone** (~5 ms) every camera frame → a 55,296-float feature vector
- **full** (~115 ms) every 32 frames → periodicity/period signals over a 64-frame window

Running the full model inline in the Vision Camera frame worklet blocks the camera
thread for ~115 ms once per second, dropping ~8% of backbone frames (measured). The
backbone tensor cannot be buffered on another JS thread without crossing the
worklet→JS bridge every frame (the original perf problem), and
react-native-worklets-core cannot share an ArrayBuffer across threads.

## Approach

A native `JumpProcessor` JSI `HostObject`. HostObjects are shared **by reference**
across worklet contexts, so the camera worklet can call it directly (same thread, no
serialization), and it can own a background `std::thread` for the full model — the same
pattern the existing async `run()` already uses (`std::async` + `CallInvoker`).

**Boundary (decided with the app owner): native owns the ring + full-model inference;
counting stays in the JS core kernels (already unit-tested).** Native delivers only the
tiny raw output (2 × bufferSize floats) per cycle.

### Data flow

```
Camera worklet thread                    JumpProcessor bg thread        JS thread
─────────────────────                    ───────────────────────        ─────────
backbone.runSync(frame)  ─┐
jumpProcessor.pushBackbone(out, n) ──► copy into native ring (no JS)
   (returns immediately)  ─┘            every fullModelInterval & idle:
                                          gather window → TfLiteInterpreterInvoke
                                          extract periodicities/periods
                                          callInvoker.invokeAsync ──────► onResult(per[], prd[], writeIdx, times)
                                                                          → JS core: average + count
```

- `pushBackbone` copies `outputSize` floats into the native ring on the **calling**
  (camera) thread — a `memcpy`, microseconds, no inference, no bridge. Returns at once.
- When `frameCount % fullModelInterval == 0` and no run is in flight, the ring window is
  snapshotted (gather/copy under a mutex) and handed to the bg thread; the camera thread
  continues immediately.
- The bg thread runs the full model and delivers the small result to JS via
  `CallInvoker::invokeAsync` (same mechanism as async `run()`).

The 55K backbone vector lives only in native memory + the one `pushBackbone` argument
(consumed synchronously). The big gathered buffer never crosses any thread boundary in
JS — it's a native→native copy.

## JSI surface

Installed alongside `__loadTensorflowModel`:

```
global.__createJumpProcessor(
  fullModel: TensorflowModel,   // the HostObject from loadTensorflowModel
  config: { bufferSize, outputSize, fullModelInterval,
            outputTensorPeriodicity, outputTensorPeriod,
            outputTensorMarks?, outputTensorEventType?, marksThreshold? },
  onResult: (periodicities: Float32Array, periods: Float32Array,
             writeIdx: number, gatherTimeMs: number, inferenceTimeMs: number,
             oldestFrameNumber: number, marks: Float32Array, eventTypes: Float32Array,
             periodicitiesRope: Float32Array, periodsRope: Float32Array,
             marksRope: Float32Array) => void
): JumpProcessor
```

### Output-head layout

Every per-slot head is a slot-major `[1, N, C]` tensor — `value[slot * C + channel]`,
`N` = `bufferSize`. `C` is inferred at read time (tensor floats / `bufferSize`), so a
head that gains channels needs no native change. The signal heads (periodicity, period,
marks) carry **feet on channel 0 and rope on channel 1**; both are delivered — the
unsuffixed arrays are feet, the `*Rope` arrays rope (empty when the head is
single-channel). The event-type head is `C` class logits per slot, reduced natively to a
per-slot argmax so the logits never cross to JS.

`JumpProcessor` methods (all callable from a worklet):
- `pushBackbone(output: Float32Array, frameNumber: number): void` — store one frame; may
  trigger a bg full-model run. Worklet-safe, returns immediately.
- `reset(): void` — clear ring + counters; abort any in-flight semantics (next result
  from a stale run is dropped via a generation counter).
- `dispose(): void` — stop the bg thread and release. Idempotent.

`onResult` is wrapped so it is always invoked on the JS thread (via CallInvoker), so the
JS callback can safely touch React state / the JS core kernels.

### Reaching the full model's interpreter

`JumpProcessor` needs the `TfLiteInterpreter*` of the full model. Rather than re-load the
model, `__createJumpProcessor` accepts the existing `TensorflowModel` HostObject and
`TensorflowPlugin` exposes its interpreter to `JumpProcessor` (friend / internal getter).
This reuses the already-loaded, already-delegated interpreter (GPU/Metal) — no second
load, no extra GPU memory.

## Threading & lifecycle

- One bg worker thread per processor, created lazily, joined on `dispose`/destructor.
- A `std::mutex` guards the ring during gather-snapshot; `pushBackbone`'s memcpy also
  takes it (cheap). Inference runs **outside** the lock on the snapshot copy.
- `std::atomic<bool> _busy` ensures only one full-model run in flight (drop-if-busy, like
  Vision Camera's runAsync).
- `std::atomic<uint64_t> _generation` bumped on reset; a result computed against an old
  generation is discarded before delivery.
- The full model interpreter is **only** invoked from the bg thread, so there's no
  contention with the camera thread's backbone runs (separate interpreters).

## Testing

- **Native correctness** of ring/gather lives in the JS core kernels already (the gather
  + averaging + counting are pure-JS, unit-tested). The native ring mirrors that exact
  index math; a focused C++ assertion test (host-compilable) covers the ring gather.
- **JS binding**: fork jest tests mock `global.__createJumpProcessor` and assert the
  TS wrapper (`createJumpProcessor`) validates args, wires `onResult`, and forwards
  `pushBackbone`/`reset`/`dispose`.
- **On-device**: the Wallaby v2 dev screen reports effective FPS + dropped %. Success =
  drops →~0 with the full model native, counts unchanged vs. the JS-core golden behavior.
```
