//
//  TensorflowPlugin.h
//  VisionCamera
//
//  Created by Marc Rousavy on 26.06.23.
//  Copyright © 2023 mrousavy. All rights reserved.
//

#pragma once

#include "jsi/TypedArray.h"
#include <jsi/jsi.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef ANDROID
#include <ReactCommon/CallInvoker.h>
#include <tflite/c/c_api.h>
#else
#include <React-callinvoker/ReactCommon/CallInvoker.h>
#include <TensorFlowLiteC/TensorFlowLiteC.h>
#endif

using namespace facebook;
using namespace mrousavy;

struct Buffer {
  void* data;
  size_t size;
};
typedef std::function<Buffer(std::string)> FetchURLFunc;

/**
 * Per-tensor debug information collected during a single inference run.
 * Each entry corresponds to one compute tensor (excludes weights/constants).
 */
struct TensorDebugInfo {
  /** Absolute tensor index in the model graph */
  int index;
  /** Tensor name, typically the name of the op that produced it */
  std::string name;
  /** Tensor shape dimensions */
  std::vector<int> shape;
  /**
   * Hardware that processed this tensor.
   * Values: "cpu", "metal", "core-ml", "nnapi", "android-gpu"
   * A value of "cpu" for an intermediate tensor when a GPU delegate is
   * configured indicates that op fell back to CPU (unsupported by delegate).
   */
  std::string hardware;
};

/**
 * Per-op timing from the TFLite telemetry profiler.
 * Only populated on iOS (TensorFlowLiteC 2.17+ includes profiler.h).
 */
struct OpTiming {
  /** Operator name, e.g. "CONV_2D", "DEPTHWISE_CONV_2D" */
  std::string name;
  /** Operator index in the model graph's execution plan */
  int64_t opIdx;
  /** Wall-clock execution time for this op in milliseconds */
  double durationMs;
};

/**
 * Aggregated statistics from a single inference run.
 * Available via model.stats after each run() / runSync() when debug mode is on.
 */
struct InferenceStats {
  /** Total wall-clock time for TfLiteInterpreterInvoke() in milliseconds */
  double totalTimeMs;
  /**
   * Hardware placement for input and output compute tensors.
   * "cpu" when a GPU delegate is active indicates a CPU fallback.
   */
  std::vector<TensorDebugInfo> tensors;
  /**
   * Per-op execution times collected via the TFLite telemetry profiler.
   * Empty on Android (profiler.h not available in LiteRT headers).
   */
  std::vector<OpTiming> opTimings;
};

// Forward declaration — full definition is in TensorflowPlugin.mm so that
// profiler.h (iOS-only) doesn't need to be included in this header.
struct TensorflowProfilerState;

class TensorflowPlugin : public jsi::HostObject {
public:
  // TFL Delegate Type
  enum Delegate { Default, Metal, CoreML, NnApi, AndroidGPU };

public:
  explicit TensorflowPlugin(TfLiteInterpreter* interpreter, Buffer model, Delegate delegate,
                            std::shared_ptr<react::CallInvoker> callInvoker,
                            bool debugMode = false,
                            std::shared_ptr<TensorflowProfilerState> profilerState = nullptr);
  ~TensorflowPlugin();

  jsi::Value get(jsi::Runtime& runtime, const jsi::PropNameID& name) override;
  std::vector<jsi::PropNameID> getPropertyNames(jsi::Runtime& runtime) override;

  static void installToRuntime(jsi::Runtime& runtime,
                               std::shared_ptr<react::CallInvoker> callInvoker,
                               FetchURLFunc fetchURL);

  /**
   * Internal accessor for the underlying TFLite interpreter. Used by
   * JumpProcessor so it can run this already-loaded (and already
   * delegate-configured) model on its own background thread without re-loading
   * the model or allocating a second interpreter / GPU context.
   *
   * Ownership stays with this TensorflowPlugin; callers must keep the plugin
   * alive for as long as they use the interpreter (JumpProcessor holds a
   * shared_ptr to the plugin to guarantee this).
   */
  TfLiteInterpreter* getInterpreter() const { return _interpreter; }

  /** Hardware the configured delegate runs on
   * ("metal", "core-ml", "nnapi", "android-gpu", or "cpu"). */
  std::string delegateHardwareName() const { return getDelegateHardwareName(); }

  /**
   * Scan the model's input/output compute tensors and report their hardware
   * placement (cpu vs the active delegate). A "cpu" tensor while a GPU delegate
   * is configured hints at a CPU fallback. Cheap; call after at least one run().
   *
   * Exposed so JumpProcessor (which runs the full model on its own thread,
   * outside TensorflowPlugin::run()) can surface placement in its benchmark stats.
   */
  std::vector<TensorDebugInfo> inspectIOTensors() const;

private:
  void copyInputBuffers(jsi::Runtime& runtime, jsi::Object inputValues);
  void run();
  jsi::Value copyOutputBuffers(jsi::Runtime& runtime);

  std::shared_ptr<TypedArrayBase> getOutputArrayForTensor(jsi::Runtime& runtime,
                                                          const TfLiteTensor* tensor);

  InferenceStats collectDebugStats(double totalMs);
  void printDebugStats(const InferenceStats& stats);
  std::string getDelegateHardwareName() const;

  // Telemetry profiler callbacks (iOS only — no-ops on Android)
#ifndef ANDROID
  static uint32_t profilerBeginOp(struct TfLiteTelemetryProfilerStruct* profiler,
                                   const char* op_name, int64_t op_idx, int64_t subgraph_idx);
  static void profilerEndOp(struct TfLiteTelemetryProfilerStruct* profiler,
                             uint32_t event_handle);
  static void profilerOpEvent(struct TfLiteTelemetryProfilerStruct* profiler,
                               const char* op_name, uint64_t elapsed_us,
                               int64_t op_idx, int64_t subgraph_idx);
#endif

private:
  TfLiteInterpreter* _interpreter = nullptr;
  Delegate _delegate = Delegate::Default;
  Buffer _model;
  std::shared_ptr<react::CallInvoker> _callInvoker;
  bool _debugMode = false;
  bool _hasStats = false;
  InferenceStats _lastStats;
  std::shared_ptr<TensorflowProfilerState> _profilerState;

  std::unordered_map<std::string, std::shared_ptr<TypedArrayBase>> _outputBuffers;
};
