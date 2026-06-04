import type { TensorflowModel } from './TensorflowLite'

/**
 * Native jump-detection processor.
 *
 * Owns a ring buffer of backbone outputs and runs the (heavy) full model on a
 * dedicated native background thread, so the camera/worklet thread is never
 * blocked by inference. Backbone outputs are pushed in via {@linkcode pushBackbone}
 * (a cheap native memcpy — the large feature vector never crosses to the JS
 * thread), and results are delivered to the `onResult` callback on the JS thread.
 *
 * Counting/averaging stays in JS (see the app's jump-detection core kernels) —
 * this processor only handles buffering + the full-model inference.
 */
export interface JumpProcessor {
  /**
   * Store one backbone output frame into the native ring. Safe to call from a
   * Vision Camera frame worklet; returns immediately. Every
   * `fullModelInterval` frames (once the ring is full) it triggers a full-model
   * run on the background thread, unless one is already in flight.
   *
   * @param output The backbone output (length must be >= `outputSize`).
   * @param frameNumber Monotonic global frame index (reserved; currently advisory).
   */
  pushBackbone(output: Float32Array, frameNumber: number): void
  /** Clear the ring + counters. Any in-flight result is discarded. */
  reset(): void
  /** Stop the background thread and release native resources. Idempotent. */
  dispose(): void
  /**
   * Debug counters for diagnosing the pipeline. Read-only snapshot; cheap to
   * poll. `lastSkip` names why the most recent run early-returned:
   * 0=ok, 1=reset/generation, 2=input-tensor-null, 3=input-copy-failed,
   * 4=invoke-failed, 5=output-tensor-null.
   */
  readonly stats: {
    framesPushed: number
    runsTriggered: number
    runsStarted: number
    runsCompleted: number
    lastSkip: number
    inputTensorBytes: number
    snapshotBytes: number
    busy: boolean
  }
}

/** Config the native processor needs (subset of the app's processor config). */
export interface JumpProcessorConfig {
  /** Ring size in frames (e.g. 64). */
  bufferSize: number
  /** Floats per backbone output frame (e.g. 55296). */
  outputSize: number
  /** Run the full model every N frames (e.g. 32). */
  fullModelInterval: number
  /** Index of the periodicity output tensor on the full model. */
  outputTensorPeriodicity: number
  /** Index of the period output tensor on the full model. */
  outputTensorPeriod: number
  /**
   * Index of the per-frame "marks" output tensor ([1,N,1]). Omit or set < 0 when
   * the model has no marks output (marks are then delivered as an empty array
   * and the consumer falls back to period-only counting).
   */
  outputTensorMarks?: number
  /** Peak-height threshold for mark detection. Default 0.1. (Advisory; the
   * consumer applies peak-finding — native only extracts the raw signal.) */
  marksThreshold?: number
}

/**
 * Called on the JS thread once per full-model run with the raw per-slot output.
 * The app's JS core kernels turn these into averaged signals + a jump count.
 *
 * @param periodicities Per-slot periodicity logits (length `bufferSize`).
 * @param periods Per-slot periods (length `bufferSize`).
 * @param writeIdx Ring write index at the time of the run (oldest-slot offset).
 * @param gatherTimeMs Time spent snapshotting the ring window (native).
 * @param inferenceTimeMs Time spent in `TfLiteInterpreterInvoke` (native).
 * @param oldestFrameNumber Absolute global frame index of slot 0 (the oldest
 *   frame) of this window. Lets the consumer reset per-slot averaging state
 *   when a ring slot now holds a different frame than it did last run.
 * @param marks Per-slot "marks" signal (length `bufferSize`), or an empty array
 *   when the model has no marks output (`outputTensorMarks` unset). Peaks in
 *   this signal are used as the primary rep-counting cue.
 */
export type JumpProcessorOnResult = (
  periodicities: Float32Array,
  periods: Float32Array,
  writeIdx: number,
  gatherTimeMs: number,
  inferenceTimeMs: number,
  oldestFrameNumber: number,
  marks: Float32Array
) => void

declare global {
  // eslint-disable-next-line no-var
  var __createJumpProcessor:
    | ((
        fullModel: TensorflowModel,
        config: JumpProcessorConfig,
        onResult: JumpProcessorOnResult
      ) => JumpProcessor)
    | undefined
  /**
   * Native build marker, set by the native installer. Lets JS verify the device
   * is running the expected native binary (dev-pod / NDK caches can link stale
   * object files even after a full rebuild). `undefined` means the native code
   * predates this marker (i.e. a stale build) or the bindings aren't installed.
   */
  // eslint-disable-next-line no-var
  var __jumpProcessorNativeBuild: string | undefined
}

/**
 * The native build marker compiled into the running binary, or `undefined` if
 * the native code is stale / not installed. Use to confirm a rebuild took.
 */
export function getJumpProcessorNativeBuild(): string | undefined {
  return global.__jumpProcessorNativeBuild
}

/**
 * Create a native {@linkcode JumpProcessor} for the given (already-loaded) full
 * model. The model's native interpreter is reused on the background thread — no
 * second load, no extra GPU context.
 *
 * @throws if the native bindings are not installed, or the model/config are invalid.
 */
export function createJumpProcessor(
  fullModel: TensorflowModel,
  config: JumpProcessorConfig,
  onResult: JumpProcessorOnResult
): JumpProcessor {
  const create = global.__createJumpProcessor
  if (create == null) {
    throw new Error(
      'react-native-fast-tflite: __createJumpProcessor is not installed. ' +
        'Rebuild the native app after upgrading the package.'
    )
  }
  if (
    config.bufferSize <= 0 ||
    config.outputSize <= 0 ||
    config.fullModelInterval <= 0
  ) {
    throw new Error(
      'createJumpProcessor: bufferSize, outputSize and fullModelInterval must all be > 0'
    )
  }
  return create(fullModel, config, onResult)
}
