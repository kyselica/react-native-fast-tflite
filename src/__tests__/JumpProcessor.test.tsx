/**
 * Tests for the createJumpProcessor JS binding.
 *
 * The native HostObject (global.__createJumpProcessor) is mocked; these verify
 * the TS wrapper validates inputs, surfaces a clear error when the native
 * binding is missing, and forwards model/config/onResult unchanged.
 */
import {
  createJumpProcessor,
  type JumpProcessor,
  type JumpProcessorConfig,
} from '../JumpProcessor'
import type { TensorflowModel } from '../TensorflowLite'

jest.mock('react-native', () => ({
  Image: { resolveAssetSource: jest.fn() },
  TurboModuleRegistry: {
    getEnforcing: jest.fn(() => ({ install: jest.fn(() => true) })),
  },
}))

const FAKE_MODEL = {
  delegate: 'metal',
  inputs: [],
  outputs: [],
  stats: undefined,
  run: jest.fn(),
  runSync: jest.fn(),
} as unknown as TensorflowModel

const VALID_CONFIG: JumpProcessorConfig = {
  bufferSize: 64,
  outputSize: 55296,
  fullModelInterval: 32,
  outputTensorPeriodicity: 0,
  outputTensorPeriod: 1,
}

/** Shared no-op onResult for cases that don't assert on the callback. */
const noop: () => void = () => undefined

function makeFakeProcessor(): JumpProcessor {
  return {
    pushBackbone: jest.fn(),
    setFullModelInterval: jest.fn(),
    reset: jest.fn(),
    dispose: jest.fn(),
    stats: {
      framesPushed: 0,
      runsTriggered: 0,
      runsStarted: 0,
      runsCompleted: 0,
      lastSkip: 0,
      inputTensorBytes: 0,
      snapshotBytes: 0,
      busy: false,
    },
  }
}

afterEach(() => {
  delete (global as any).__createJumpProcessor
  jest.restoreAllMocks()
})

describe('createJumpProcessor', () => {
  it('throws a clear error when the native binding is missing', () => {
    expect(() => createJumpProcessor(FAKE_MODEL, VALID_CONFIG, noop)).toThrow(
      /__createJumpProcessor is not installed/
    )
  })

  it('forwards model, config and onResult to the native factory', () => {
    const fake = makeFakeProcessor()
    const native = jest.fn(() => fake)
    ;(global as any).__createJumpProcessor = native

    const onResult = jest.fn()
    const proc = createJumpProcessor(FAKE_MODEL, VALID_CONFIG, onResult)

    expect(native).toHaveBeenCalledTimes(1)
    expect(native).toHaveBeenCalledWith(FAKE_MODEL, VALID_CONFIG, onResult)
    expect(proc).toBe(fake)
  })

  it.each([
    ['bufferSize', { ...VALID_CONFIG, bufferSize: 0 }],
    ['outputSize', { ...VALID_CONFIG, outputSize: 0 }],
    ['fullModelInterval', { ...VALID_CONFIG, fullModelInterval: 0 }],
  ])('rejects non-positive %s before calling native', (_label, badConfig) => {
    const native = jest.fn()
    ;(global as any).__createJumpProcessor = native
    expect(() => createJumpProcessor(FAKE_MODEL, badConfig, noop)).toThrow(
      /must all be > 0/
    )
    expect(native).not.toHaveBeenCalled()
  })

  it('returns a processor whose methods are callable', () => {
    const fake = makeFakeProcessor()
    ;(global as any).__createJumpProcessor = jest.fn(() => fake)

    const proc = createJumpProcessor(FAKE_MODEL, VALID_CONFIG, noop)
    const frame = new Float32Array(VALID_CONFIG.outputSize)
    proc.pushBackbone(frame, 0)
    proc.reset()
    proc.dispose()

    expect(fake.pushBackbone).toHaveBeenCalledWith(frame, 0)
    expect(fake.reset).toHaveBeenCalledTimes(1)
    expect(fake.dispose).toHaveBeenCalledTimes(1)
  })
})
