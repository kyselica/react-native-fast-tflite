/**
 * Tests for react-native-fast-tflite
 *
 * These tests mock the JSI layer (global.__loadTensorflowModel) and verify:
 *  1. loadTensorflowModel resolves to the native model when debugMode is off
 *  2. loadTensorflowModel resolves to a working wrapper when debugMode is on
 *  3. The wrapper calls native run/runSync and returns the real outputs
 *  4. model.stats is populated after each inference in debug mode
 *  5. console.log is called with the expected timing/hardware lines after inference
 *  6. No extra console.log calls happen when debugMode is off
 */

import { Image } from 'react-native'
import {
  loadTensorflowModel,
  type TensorflowModel,
  type InferenceStats,
} from '../TensorflowLite'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const FAKE_OUTPUT = new Float32Array([0.1, 0.9])

function makeFakeStats(overrides: Partial<InferenceStats> = {}): InferenceStats {
  return {
    totalTimeMs: 12.345,
    tensors: [
      { index: 0, name: 'input', shape: [1, 4], hardware: 'cpu' },
      { index: 3, name: 'dense/BiasAdd', shape: [1, 2], hardware: 'cpu' },
    ],
    opTimings: [
      { name: 'CONV_2D', opIdx: 0, durationMs: 4.321 },
      { name: 'SOFTMAX', opIdx: 1, durationMs: 0.123 },
    ],
    ...overrides,
  }
}

/** Build a minimal TensorflowModel mock */
function makeFakeNativeModel(options: {
  delegate?: TensorflowModel['delegate']
  stats?: InferenceStats | undefined
  runSyncImpl?: (input: unknown[]) => unknown[]
  runImpl?: (input: unknown[]) => Promise<unknown[]>
} = {}): TensorflowModel {
  let lastStats: InferenceStats | undefined = options.stats

  const model: TensorflowModel = {
    delegate: options.delegate ?? 'default',
    inputs: [{ name: 'input', dataType: 'float32', shape: [1, 4] }],
    outputs: [{ name: 'output', dataType: 'float32', shape: [1, 2] }],
    get stats() {
      return lastStats
    },
    runSync(input: unknown[]) {
      const result = options.runSyncImpl ? options.runSyncImpl(input) : [FAKE_OUTPUT]
      // Simulate native side populating stats after invoke
      lastStats = makeFakeStats()
      return result as ReturnType<TensorflowModel['runSync']>
    },
    async run(input: unknown[]) {
      const result = options.runImpl
        ? await options.runImpl(input)
        : [FAKE_OUTPUT]
      lastStats = makeFakeStats()
      return result as Awaited<ReturnType<TensorflowModel['run']>>
    },
  }
  return model
}

// ---------------------------------------------------------------------------
// Module-level setup
// ---------------------------------------------------------------------------

jest.mock('react-native', () => ({
  Image: {
    resolveAssetSource: jest.fn(() => ({
      uri: 'file:///fake/model.tflite',
    })),
  },
  TurboModuleRegistry: {
    getEnforcing: jest.fn(() => ({ install: jest.fn(() => true) })),
  },
}))

let fakeNativeModel: TensorflowModel

beforeEach(() => {
  fakeNativeModel = makeFakeNativeModel()
  ;(global as any).__loadTensorflowModel = jest.fn(() =>
    Promise.resolve(fakeNativeModel)
  )
  jest.spyOn(console, 'log').mockImplementation(() => {})
})

afterEach(() => {
  jest.restoreAllMocks()
  delete (global as any).__loadTensorflowModel
})

// ---------------------------------------------------------------------------
// loadTensorflowModel — basic resolution
// ---------------------------------------------------------------------------

describe('loadTensorflowModel', () => {
  it('resolves with the native model when debugMode is off (default)', async () => {
    const model = await loadTensorflowModel({ url: 'file:///model.tflite' })
    expect(model).toBe(fakeNativeModel)
    expect((global as any).__loadTensorflowModel).toHaveBeenCalledWith(
      'file:///model.tflite',
      'default',
      1,
      false,
      false
    )
  })

  it('resolves with the native model when debugMode is explicitly false', async () => {
    const model = await loadTensorflowModel(
      { url: 'file:///model.tflite' },
      { enableDebugMode: false }
    )
    expect(model).toBe(fakeNativeModel)
  })

  it('resolves a require() asset via Image.resolveAssetSource', async () => {
    const model = await loadTensorflowModel(1 as any)
    expect(Image.resolveAssetSource).toHaveBeenCalledWith(1)
    expect((global as any).__loadTensorflowModel).toHaveBeenCalledWith(
      'file:///fake/model.tflite',
      'default',
      1,
      false,
      false
    )
    expect(model).toBe(fakeNativeModel)
  })

  it('forwards delegate and numThreads options', async () => {
    await loadTensorflowModel(
      { url: 'file:///model.tflite' },
      { delegate: 'metal', numThreads: 4 }
    )
    expect((global as any).__loadTensorflowModel).toHaveBeenCalledWith(
      'file:///model.tflite',
      'metal',
      4,
      false,
      false
    )
  })

  it('forwards enableDebugMode: true as the fourth argument', async () => {
    await loadTensorflowModel(
      { url: 'file:///model.tflite' },
      { enableDebugMode: true }
    )
    expect((global as any).__loadTensorflowModel).toHaveBeenCalledWith(
      'file:///model.tflite',
      'default',
      1,
      true,
      false
    )
  })

  it('forwards enableFp16: true as the fifth argument', async () => {
    await loadTensorflowModel(
      { url: 'file:///model.tflite' },
      { enableFp16: true }
    )
    expect((global as any).__loadTensorflowModel).toHaveBeenCalledWith(
      'file:///model.tflite',
      'default',
      1,
      false,
      true
    )
  })

  it('returns a wrapper (not the native model) when debugMode is on', async () => {
    const model = await loadTensorflowModel(
      { url: 'file:///model.tflite' },
      { enableDebugMode: true }
    )
    // The wrapper is a plain object, not the same reference as the native model
    expect(model).not.toBe(fakeNativeModel)
  })
})

// ---------------------------------------------------------------------------
// Debug wrapper — inference still works
// ---------------------------------------------------------------------------

describe('debug wrapper — runSync', () => {
  it('returns the same output as the native model', async () => {
    const model = await loadTensorflowModel(
      { url: 'file:///model.tflite' },
      { enableDebugMode: true }
    )
    const input = new Float32Array([1, 2, 3, 4])
    const output = model.runSync([input])
    expect(output).toEqual([FAKE_OUTPUT])
  })

  it('exposes the same delegate, inputs, and outputs as the native model', async () => {
    fakeNativeModel = makeFakeNativeModel({ delegate: 'metal' })
    ;(global as any).__loadTensorflowModel = jest.fn(() =>
      Promise.resolve(fakeNativeModel)
    )
    const model = await loadTensorflowModel(
      { url: 'file:///model.tflite' },
      { enableDebugMode: true }
    )
    expect(model.delegate).toBe('metal')
    expect(model.inputs).toEqual(fakeNativeModel.inputs)
    expect(model.outputs).toEqual(fakeNativeModel.outputs)
  })

  it('populates model.stats after runSync', async () => {
    const model = await loadTensorflowModel(
      { url: 'file:///model.tflite' },
      { enableDebugMode: true }
    )
    expect(model.stats).toBeUndefined()
    model.runSync([new Float32Array([1, 2, 3, 4])])
    expect(model.stats).toBeDefined()
    expect(model.stats!.totalTimeMs).toBe(12.345)
    expect(model.stats!.tensors).toHaveLength(2)
  })

  it('logs a summary line and one line per tensor to console after runSync', async () => {
    const model = await loadTensorflowModel(
      { url: 'file:///model.tflite' },
      { enableDebugMode: true }
    )
    model.runSync([new Float32Array([1, 2, 3, 4])])

    const logs = (console.log as jest.Mock).mock.calls.map((c) => c[0])
    const tfliteLogs = logs.filter((l) => typeof l === 'string' && l.startsWith('[TFLite]'))

    // Summary line
    expect(tfliteLogs[0]).toMatch(/Inference:.*12\.345ms/)
    expect(tfliteLogs[0]).toMatch(/delegate: default/)
    // Per-tensor lines
    expect(tfliteLogs[1]).toMatch(/\[0\].*input.*cpu/)
    expect(tfliteLogs[2]).toMatch(/\[3\].*dense\/BiasAdd.*cpu/)
  })

  it('does NOT log [TFLite] lines when debugMode is off', async () => {
    const model = await loadTensorflowModel({ url: 'file:///model.tflite' })
    model.runSync([new Float32Array([1, 2, 3, 4])])

    const logs = (console.log as jest.Mock).mock.calls.map((c) => c[0])
    const tfliteLogs = logs.filter((l) => typeof l === 'string' && l.startsWith('[TFLite]'))
    expect(tfliteLogs).toHaveLength(0)
  })
})

describe('debug wrapper — run (async)', () => {
  it('returns the same output as the native model', async () => {
    const model = await loadTensorflowModel(
      { url: 'file:///model.tflite' },
      { enableDebugMode: true }
    )
    const output = await model.run([new Float32Array([1, 2, 3, 4])])
    expect(output).toEqual([FAKE_OUTPUT])
  })

  it('populates model.stats after run resolves', async () => {
    const model = await loadTensorflowModel(
      { url: 'file:///model.tflite' },
      { enableDebugMode: true }
    )
    await model.run([new Float32Array([1, 2, 3, 4])])
    expect(model.stats).toBeDefined()
    expect(model.stats!.totalTimeMs).toBe(12.345)
  })

  it('logs after run resolves', async () => {
    const model = await loadTensorflowModel(
      { url: 'file:///model.tflite' },
      { enableDebugMode: true }
    )
    await model.run([new Float32Array([1, 2, 3, 4])])

    const logs = (console.log as jest.Mock).mock.calls.map((c) => c[0])
    const tfliteLogs = logs.filter((l) => typeof l === 'string' && l.startsWith('[TFLite]'))
    expect(tfliteLogs.length).toBeGreaterThanOrEqual(1)
    expect(tfliteLogs[0]).toMatch(/Inference:.*12\.345ms/)
  })
})

// ---------------------------------------------------------------------------
// Debug wrapper — CPU fallback warning
// ---------------------------------------------------------------------------

describe('debug wrapper — CPU fallback detection', () => {
  it('marks ⚠ CPU fallback on a cpu tensor when a GPU delegate is configured', async () => {
    fakeNativeModel = makeFakeNativeModel({
      delegate: 'metal',
      stats: undefined,
    })
    // Override runSync to report one GPU tensor and one CPU tensor
    const gpuStats: InferenceStats = {
      totalTimeMs: 3.5,
      tensors: [
        { index: 0, name: 'input', shape: [1, 4], hardware: 'cpu' },
        { index: 3, name: 'Conv1/Relu6', shape: [1, 8], hardware: 'metal' },
        { index: 7, name: 'unsupported_op', shape: [1, 2], hardware: 'cpu' },
      ],
    }
    let capturedStats: InferenceStats | undefined
    const originalRunSync = fakeNativeModel.runSync.bind(fakeNativeModel)
    ;(fakeNativeModel as any).runSync = function (input: unknown[]) {
      const result = originalRunSync(input as any)
      capturedStats = gpuStats
      return result
    }
    Object.defineProperty(fakeNativeModel, 'stats', {
      get: () => capturedStats,
    })

    ;(global as any).__loadTensorflowModel = jest.fn(() =>
      Promise.resolve(fakeNativeModel)
    )

    const model = await loadTensorflowModel(
      { url: 'file:///model.tflite' },
      { delegate: 'metal', enableDebugMode: true }
    )
    model.runSync([new Float32Array([1, 2, 3, 4])])

    const logs = (console.log as jest.Mock).mock.calls.map((c) => c[0] as string)
    const fallbackLine = logs.find((l) => l.includes('unsupported_op'))
    expect(fallbackLine).toBeDefined()
    expect(fallbackLine).toMatch(/⚠/)
    expect(fallbackLine).toMatch(/CPU fallback/)

    // GPU tensor should NOT have the fallback warning
    const gpuLine = logs.find((l) => l.includes('Conv1/Relu6'))
    expect(gpuLine).toBeDefined()
    expect(gpuLine).not.toMatch(/⚠/)
  })
})
