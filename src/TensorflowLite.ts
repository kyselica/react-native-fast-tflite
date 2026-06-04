import { useEffect, useState } from 'react'
import { Image } from 'react-native'
import { TensorflowModule } from './TensorflowModule'

type TypedArray =
  | Float32Array
  | Float64Array
  | Int8Array
  | Int16Array
  | Int32Array
  | Uint8Array
  | Uint16Array
  | Uint32Array
  | BigInt64Array
  | BigUint64Array

declare global {
  /**
   * Loads the Model into memory. Path is fetchable resource, e.g.:
   * http://192.168.8.110:8081/assets/assets/model.tflite?platform=ios&hash=32e9958c83e5db7d0d693633a9f0b175
   */
  // eslint-disable-next-line no-var
  var __loadTensorflowModel: (
    path: string,
    delegate: TensorflowModelDelegate,
    numThreads: number,
    debugMode: boolean
  ) => Promise<TensorflowModel>
}
// Installs the JSI bindings into the global namespace.
console.log('Installing bindings...')
const result = TensorflowModule.install() as boolean
if (result !== true)
  console.error('Failed to install Tensorflow Lite bindings!')

console.log('Successfully installed!')

export type TensorflowModelDelegate =
  | 'default'
  | 'metal'
  | 'core-ml'
  | 'nnapi'
  | 'android-gpu'

export interface TensorflowModelOptions {
  /**
   * The delegate to use for computations.
   * Uses the standard CPU delegate per default.
   * The `core-ml` or `metal` delegates are GPU-accelerated, but don't work on every model.
   * @default 'default'
   */
  delegate?: TensorflowModelDelegate
  /**
   * The number of threads to use for inference.
   * Uses 1 threads by default.
   * @default 1
   */
  numThreads?: number
  /**
   * Enable debug mode to log per-tensor hardware placement and inference timing
   * after each call to `run()` or `runSync()`.
   *
   * When enabled:
   * - Timing and hardware info are printed to the native console (logcat / Xcode console).
   * - `model.stats` is populated after each inference.
   * - A `⚠ CPU fallback` warning is shown for any tensor that ran on CPU while a
   *   hardware delegate (Metal, CoreML, NNAPI, Android GPU) was configured — this
   *   indicates an op that the delegate does not support.
   *
   * Has no effect on inference results or correctness. Adds a small overhead from
   * chrono timing and tensor inspection; keep disabled in production.
   * @default false
   */
  enableDebugMode?: boolean
}

/**
 * Per-tensor hardware placement info collected during one inference run.
 */
export interface TensorDebugInfo {
  /** Absolute tensor index in the model's tensor graph */
  index: number
  /** Tensor name — typically the name of the op that produced it */
  name: string
  /** Tensor shape */
  shape: number[]
  /**
   * Hardware that processed this tensor.
   * - `'cpu'` — ran on the CPU.
   * - `'metal'` — ran on the Metal GPU delegate (iOS).
   * - `'core-ml'` — ran on the CoreML delegate (iOS).
   * - `'nnapi'` — ran on the NNAPI delegate (Android).
   * - `'android-gpu'` — ran on the Android GPU delegate.
   *
   * A `'cpu'` value for an intermediate tensor when a GPU/NPU delegate is
   * configured means that op fell back to CPU because the delegate does not
   * support it.
   */
  hardware: 'cpu' | 'metal' | 'core-ml' | 'nnapi' | 'android-gpu'
}

/**
 * Per-op timing entry from the TFLite telemetry profiler.
 * Only populated on iOS (TensorFlowLiteC 2.17+ ships profiler.h).
 * Android returns an empty array.
 */
export interface OpTiming {
  /** Operator name, e.g. `"CONV_2D"`, `"DEPTHWISE_CONV_2D"` */
  name: string
  /** Operator index in the model's execution plan */
  opIdx: number
  /** Wall-clock execution time for this op in milliseconds */
  durationMs: number
}

/**
 * Aggregated statistics from a single inference run.
 * Available via `model.stats` after each `run()` / `runSync()` call when
 * `enableDebugMode: true` is set in the model options.
 */
export interface InferenceStats {
  /**
   * Wall-clock time for `TfLiteInterpreterInvoke()` in milliseconds.
   * Does not include input/output copy time.
   */
  totalTimeMs: number
  /**
   * Hardware placement for input and output compute tensors.
   * A `'cpu'` value when a GPU/NPU delegate is configured means that
   * tensor's op fell back to CPU (unsupported by the delegate).
   */
  tensors: TensorDebugInfo[]
  /**
   * Per-op execution times recorded by the TFLite telemetry profiler.
   * Populated on iOS with TensorFlowLiteC 2.17+; empty array on Android.
   * `undefined` only in test/mock contexts where the native layer is absent.
   */
  opTimings?: OpTiming[]
}

export interface Tensor {
  /**
   * The name of the Tensor.
   */
  name: string
  /**
   * The data-type all values of this Tensor are represented in.
   */
  dataType:
    | 'bool'
    | 'uint8'
    | 'int8'
    | 'int16'
    | 'int32'
    | 'int64'
    | 'float16'
    | 'float32'
    | 'float64'
    | 'invalid'
  /**
   * The shape of the data from this tensor.
   */
  shape: number[]
}

export interface TensorflowModel {
  /**
   * The computation delegate used by this Model.
   * While CoreML and Metal delegates might be faster as they use the GPU, not all models support those delegates.
   */
  delegate: TensorflowModelDelegate
  /**
   * Run the Tensorflow Model with the given input buffer.
   * The input buffer has to match the input tensor's shape.
   */
  run(input: TypedArray[]): Promise<TypedArray[]>
  /**
   * Synchronously run the Tensorflow Model with the given input buffer.
   * The input buffer has to match the input tensor's shape.
   */
  runSync(input: TypedArray[]): TypedArray[]

  /**
   * All input tensors of this Tensorflow Model.
   */
  inputs: Tensor[]
  /**
   * All output tensors of this Tensorflow Model.
   * The user is responsible for correctly interpreting this data.
   */
  outputs: Tensor[]

  /**
   * Debug statistics from the most recent inference run.
   * Only populated when `enableDebugMode: true` was set in the model options.
   * Updated after every call to `run()` or `runSync()`.
   */
  stats: InferenceStats | undefined
}

// In React Native, `require(..)` returns a number.
type Require = number // ReturnType<typeof require>
type ModelSource = Require | { url: string }

export type TensorflowPlugin =
  | {
      model: TensorflowModel
      state: 'loaded'
    }
  | {
      model: undefined
      state: 'loading'
    }
  | {
      model: undefined
      error: Error
      state: 'error'
    }

// ---------------------------------------------------------------------------
// Debug-mode helpers — all logging happens on the JS thread after inference
// completes, so there are no cross-thread JSI calls and no risk of hangs.
// ---------------------------------------------------------------------------

function printStats(model: TensorflowModel): void {
  const stats = model.stats
  if (!stats) return
  const delegateName = model.delegate

  let cpuCount = 0
  let delegateCount = 0
  for (const t of stats.tensors) {
    if (t.hardware === 'cpu') cpuCount++
    else delegateCount++
  }

  console.log(
    `[TFLite] Inference: ${stats.totalTimeMs.toFixed(3)}ms | delegate: ${delegateName} | tensors: ${cpuCount} cpu, ${delegateCount} ${delegateName}`
  )

  // Per-tensor hardware placement
  for (const t of stats.tensors) {
    const isFallback = t.hardware === 'cpu' && delegateName !== 'default'
    console.log(
      `[TFLite]   tensor[${t.index}] '${t.name}' [${t.shape.join(',')}] \u2192 ${t.hardware}${isFallback ? ' \u26A0 CPU fallback' : ''}`
    )
  }

  // Per-op timing (iOS only; empty on Android)
  if (stats.opTimings && stats.opTimings.length > 0) {
    console.log(`[TFLite] Per-op timings (${stats.opTimings.length} ops):`)
    for (const op of stats.opTimings) {
      console.log(`[TFLite]   op[${op.opIdx}] '${op.name}' ${op.durationMs.toFixed(3)}ms`)
    }
  }
}

/**
 * Wraps a native TensorflowModel to automatically print InferenceStats to the
 * Metro / JS console after every run() / runSync() call.
 * The wrapper is a plain JS object — no JSI property writes, no background
 * thread callbacks — so it cannot cause hangs.
 */
function wrapWithDebugLogging(native: TensorflowModel): TensorflowModel {
  return {
    get delegate() {
      return native.delegate
    },
    get inputs() {
      return native.inputs
    },
    get outputs() {
      return native.outputs
    },
    get stats() {
      return native.stats
    },
    runSync(input: TypedArray[]) {
      const result = native.runSync(input)
      printStats(native)
      return result
    },
    async run(input: TypedArray[]) {
      const result = await native.run(input)
      printStats(native)
      return result
    },
  }
}

/**
 * Load a Tensorflow Lite Model from the given `.tflite` asset.
 *
 * * If you are passing in a `.tflite` model from your app's bundle using `require(..)`, make sure to add `tflite` as an asset extension to `metro.config.js`!
 * * If you are passing in a `{ url: ... }`, make sure the URL points directly to a `.tflite` model. This can either be a web URL (`http://..`/`https://..`), or a local file (`file://..`).
 *
 * @param source The `.tflite` model in form of either a `require(..)` statement or a `{ url: string }`.
 * @param delegateOrOptions The delegate to use for computations (string), or an options object containing `delegate` and `numThreads`. Uses the standard CPU delegate and 1 thread per default.
 * @returns The loaded Model.
 */
export function loadTensorflowModel(
  source: ModelSource,
  delegateOrOptions:
    | TensorflowModelDelegate
    | TensorflowModelOptions = 'default'
): Promise<TensorflowModel> {
  let uri: string
  if (typeof source === 'number') {
    console.log(`Loading Tensorflow Lite Model ${source}`)
    const asset = Image.resolveAssetSource(source)
    uri = asset.uri
    console.log(`Resolved Model path: ${asset.uri}`)
  } else if (typeof source === 'object' && 'url' in source) {
    uri = source.url
  } else {
    throw new Error(
      'TFLite: Invalid source passed! Source should be either a React Native require(..) or a `{ url: string }` object!'
    )
  }

  // Parse options
  let delegate: TensorflowModelDelegate = 'default'
  let numThreads = 1
  let debugMode = false
  if (typeof delegateOrOptions === 'string') {
    delegate = delegateOrOptions
  } else if (typeof delegateOrOptions === 'object') {
    delegate = delegateOrOptions.delegate ?? 'default'
    numThreads = delegateOrOptions.numThreads ?? 1
    debugMode = delegateOrOptions.enableDebugMode ?? false
  }

  // Use .then() rather than async/await so we never introduce an extra microtask
  // boundary around the JSI HostObject — async functions can cause JSI values to
  // be accessed outside their valid scope in some RN/JSI versions.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return (global as any).__loadTensorflowModel(uri, delegate, numThreads, debugMode).then(
    (nativeModel: TensorflowModel) =>
      debugMode ? wrapWithDebugLogging(nativeModel) : nativeModel
  )
}

/**
 * Load a Tensorflow Lite Model from the given `.tflite` asset into a React State.
 *
 * * If you are passing in a `.tflite` model from your app's bundle using `require(..)`, make sure to add `tflite` as an asset extension to `metro.config.js`!
 * * If you are passing in a `{ url: ... }`, make sure the URL points directly to a `.tflite` model. This can either be a web URL (`http://..`/`https://..`), or a local file (`file://..`).
 *
 * @param source The `.tflite` model in form of either a `require(..)` statement or a `{ url: string }`.
 * @param delegateOrOptions The delegate to use for computations (string), or an options object containing `delegate` and `numThreads`. Uses the standard CPU delegate and 1 thread per default.
 * @returns The state of the Model.
 */
export function useTensorflowModel(
  source: ModelSource,
  delegateOrOptions:
    | TensorflowModelDelegate
    | TensorflowModelOptions = 'default'
): TensorflowPlugin {
  const [state, setState] = useState<TensorflowPlugin>({
    model: undefined,
    state: 'loading',
  })

  useEffect(() => {
    const load = async (): Promise<void> => {
      try {
        setState({ model: undefined, state: 'loading' })
        const m = await loadTensorflowModel(source, delegateOrOptions)
        setState({ model: m, state: 'loaded' })
        console.log('Model loaded!')
      } catch (e) {
        console.error(`Failed to load Tensorflow Model ${source}!`, e)
        setState({ model: undefined, state: 'error', error: e as Error })
      }
    }
    load()
  }, [delegateOrOptions, source])

  return state
}
