//
//  JumpProcessor.h
//  react-native-fast-tflite
//
//  Native ring buffer + background-thread full-model runner for jump detection.
//  See JumpProcessor.DESIGN.md for the rationale.
//

#pragma once

#include "TensorflowPlugin.h"
#include "jsi/TypedArray.h"
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <jsi/jsi.h>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#ifdef ANDROID
#include <ReactCommon/CallInvoker.h>
#include <tflite/c/c_api.h>
#else
#include <React-callinvoker/ReactCommon/CallInvoker.h>
#include <TensorFlowLiteC/TensorFlowLiteC.h>
#endif

using namespace facebook;

/**
 * Per-slot output-head layout. Every head of the jump models is a slot-major
 * tensor [1, N, C]: N = bufferSize slots, C channels per slot, laid out
 * `value[slot * C + channel]`. C is inferred at read time from the tensor's
 * float count / bufferSize, so a head gaining channels never needs a native
 * change — only a channel constant here (or a new config field) to read it.
 *
 * The signal heads (periodicity, period, marks) carry one channel per tracked
 * body: feet first, then rope.
 */
enum JumpSignalChannel : int {
  kJumpChannelFeet = 0,
  kJumpChannelRope = 1,
};

/**
 * Configuration for the jump-detection ring + full-model run. Mirrors the
 * subset of the JS processor config the native side needs.
 */
struct JumpProcessorConfig {
  int bufferSize = 0;              // ring slots (frames)
  int outputSize = 0;              // floats per backbone frame
  int fullModelInterval = 0;       // run full model every N frames
  int outputTensorPeriodicity = 0; // index of periodicity output tensor
  int outputTensorPeriod = 0;      // index of period output tensor
  // Index of the per-frame "marks" output tensor ([1,N,C]); -1 = model has no
  // marks output (marks then delivered as zeros; JS falls back to period-only).
  int outputTensorMarks = -1;
  // Index of the per-frame "event type" output tensor ([1,N,C], C class logits
  // per slot); -1 = model has no event-type output (event types then delivered
  // as an empty array). Native takes the per-slot argmax and delivers the class
  // index; the raw logits never cross to JS.
  int outputTensorEventType = -1;
};

/**
 * Owns a ring of backbone outputs and runs the full model on a dedicated
 * background thread. Exposed to JS as a HostObject so it can be called from a
 * Vision Camera frame worklet (HostObjects are shared by reference across
 * worklet contexts).
 *
 * pushBackbone() copies one backbone frame into the ring (cheap memcpy on the
 * calling thread) and, every fullModelInterval frames, signals the worker to
 * gather + run the full model. Results are delivered to JS via CallInvoker.
 */
class JumpProcessor : public jsi::HostObject {
public:
  JumpProcessor(std::shared_ptr<TensorflowPlugin> fullModel, JumpProcessorConfig config,
                std::shared_ptr<react::CallInvoker> callInvoker,
                std::shared_ptr<jsi::Function> onResult, jsi::Runtime& runtime);
  ~JumpProcessor() override;

  jsi::Value get(jsi::Runtime& runtime, const jsi::PropNameID& name) override;
  std::vector<jsi::PropNameID> getPropertyNames(jsi::Runtime& runtime) override;

  /** Install global.__createJumpProcessor into the runtime. */
  static void installToRuntime(jsi::Runtime& runtime,
                               std::shared_ptr<react::CallInvoker> callInvoker);

private:
  void pushBackbone(const float* data, size_t length, int64_t frameNumber);
  void reset();
  void dispose();
  // Change how often the full model runs (every N pushed frames). Atomic so it
  // can be flipped from the JS thread at any time; the trigger check reads it.
  // Clamped to >= 1; no effect on the ring (the model still consumes bufferSize
  // frames each run), only on the trigger cadence.
  void setFullModelInterval(int interval);

  void workerLoop();
  // Runs on the worker thread: invoke the full model on the pre-gathered
  // _snapshot, extract periodicity/period, deliver to JS.
  // `oldestFrameNumber` is the absolute global frame index of slot 0 of the
  // window — JS uses it to reset per-slot averaging when a slot's frame changes.
  void runFullModelOnWorker(uint64_t generation, int writeIdx, int64_t oldestFrameNumber);

  std::shared_ptr<TensorflowPlugin> _fullModel; // keeps interpreter alive
  TfLiteInterpreter* _interpreter = nullptr;
  JumpProcessorConfig _config;
  std::shared_ptr<react::CallInvoker> _callInvoker;
  std::shared_ptr<jsi::Function> _onResult;
  jsi::Runtime& _runtime;

  // Ring storage (native-only; never crosses a thread boundary as JS).
  std::vector<float> _ring;     // bufferSize * outputSize
  std::vector<float> _snapshot; // gathered window handed to the worker
  std::mutex _ringMutex;

  int _writeIndex = 0;
  int64_t _frameCount = 0;
  // Run the full model every N frames. Seeded from config.fullModelInterval;
  // mutable at runtime via setFullModelInterval(). Atomic: written on the JS
  // thread, read in pushBackbone (camera thread).
  std::atomic<int> _fullModelInterval{0};

  // Worker thread + signalling
  std::thread _worker;
  std::mutex _workMutex;
  std::condition_variable _workCv;
  bool _workPending = false;
  int _pendingWriteIdx = 0;
  int64_t _pendingOldestFrame = 0;
  uint64_t _pendingGeneration = 0;
  std::atomic<bool> _busy{false};
  std::atomic<bool> _stop{false};
  std::atomic<uint64_t> _generation{0};

  // Debug counters (observability for the dev screen — see the `stats` prop).
  // All atomic so they can be read from the JS thread while the worker writes.
  std::atomic<uint64_t> _framesPushed{0};  // pushBackbone calls
  std::atomic<uint64_t> _runsTriggered{0}; // full-model runs scheduled
  std::atomic<uint64_t> _runsStarted{0};   // runFullModelOnWorker entered
  std::atomic<uint64_t> _runsCompleted{0}; // results delivered to JS
  // Why the last run early-returned (0=ok, 1=reset/generation, 2=input-null,
  // 3=copy-failed, 4=invoke-failed, 5=output-null, 6=deliver-threw). See
  // SkipReason in the .mm.
  std::atomic<int> _lastSkip{0};
  std::atomic<int> _lastInputBytes{0}; // input tensor byte size (debug)
  std::atomic<int> _expectedBytes{0};  // snapshot byte size (debug)

  // Sub-step wall-clock timings (milliseconds) of the most recent full-model
  // run, for the benchmark dev screen. Always measured — a handful of clock
  // reads on a path that runs ~once/second, so negligible in production. Atomic
  // so the JS thread can read them via `stats` while the worker writes.
  //   gather  — snapshotting the ring window (camera thread, in pushBackbone)
  //   copyIn  — TfLiteTensorCopyFromBuffer of the snapshot into the input tensor
  //   invoke  — TfLiteInterpreterInvoke (the actual model compute)
  //   copyOut — reading periodicity/period/marks output tensors back out
  std::atomic<double> _lastGatherMs{0.0};
  std::atomic<double> _lastCopyInMs{0.0};
  std::atomic<double> _lastInvokeMs{0.0};
  std::atomic<double> _lastCopyOutMs{0.0};

  // I/O tensor hardware placement (cpu vs delegate), captured once on the worker
  // thread after the first successful invoke (placement is stable run-to-run).
  // Guarded by _placementMutex; read by the `stats` getter on the JS thread.
  mutable std::mutex _placementMutex;
  std::vector<TensorDebugInfo> _placement;
  std::atomic<bool> _hasPlacement{false};
};
