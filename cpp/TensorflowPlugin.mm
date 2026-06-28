//
//  TensorflowPlugin.m
//  VisionCamera
//
//  Created by Marc Rousavy on 26.06.23.
//  Copyright © 2023 mrousavy. All rights reserved.
//

#include "TensorflowPlugin.h"

#include "TensorHelpers.h"
#include "jsi/Promise.h"
#include "jsi/TypedArray.h"
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstring>
#include <future>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>

#ifdef ANDROID
#include <android/log.h>
#include <tflite/c/c_api.h>
#include <tflite/delegates/gpu/delegate.h>
#include <tflite/delegates/nnapi/nnapi_delegate_c_api.h>

// XNNPACK delegate (CPU). The symbols are exported by libtensorflowlite_jni.so,
// but litert 1.4.0 does not ship xnnpack_delegate.h, so we declare the minimal
// C API here. The struct is byte-identical across TF 2.18/2.19 (the litert 1.4.0
// lineage); only the leading fields are read/written. `_reserved` over-allocates
// the struct so that if this litert build appends trailing fields,
// TfLiteXNNPackDelegateOptionsDefault() writes into the padding instead of
// overflowing our stack slot.
extern "C" {
typedef struct {
  int32_t num_threads;
  uint32_t flags;
  void* weights_cache;
  bool handle_variable_ops;
  const char* weight_cache_file_path;
  uint8_t _reserved[64];
} FastTfliteXNNPackDelegateOptions;
FastTfliteXNNPackDelegateOptions TfLiteXNNPackDelegateOptionsDefault(void);
TfLiteDelegate* TfLiteXNNPackDelegateCreate(const FastTfliteXNNPackDelegateOptions* options);
void TfLiteXNNPackDelegateDelete(TfLiteDelegate* delegate);
}
#define FAST_TFLITE_XNNPACK_FLAG_FORCE_FP16 0x00000004u
#else
#include <TensorFlowLiteC/TensorFlowLiteC.h>
#include <TensorFlowLiteC/c_api_experimental.h>
#include <TensorFlowLiteC/profiler.h>

// Delegate wrappers are implemented in ios/DelegateWrapper.mm (Objective-C++)
// because the delegate framework headers may import Objective-C frameworks.
// Forward declare the C functions here - the implementation handles availability checks.

// Metal delegate
extern "C" TfLiteDelegate* TFLCreateMetalDelegate(bool allowPrecisionLoss);
extern "C" void TFLDeleteMetalDelegate(TfLiteDelegate* delegate);
extern "C" bool TFLIsMetalDelegateAvailable(void);

// CoreML delegate
extern "C" TfLiteDelegate* TFLCreateCoreMLDelegate(void);
extern "C" void TFLDeleteCoreMLDelegate(TfLiteDelegate* delegate);
extern "C" bool TFLIsCoreMLDelegateAvailable(void);
#endif

using namespace facebook;
using namespace mrousavy;

static void log(const char* format, ...) {
  va_list args;
  va_start(args, format);
#ifdef ANDROID
  __android_log_vprint(ANDROID_LOG_DEBUG, "RNFastTflite", format, args);
#else
  vprintf(format, args);
  printf("\n");
#endif
  va_end(args);
}

// ---------------------------------------------------------------------------
// Per-op profiler state — iOS only (profiler.h not available in Android LiteRT)
// ---------------------------------------------------------------------------

struct TensorflowProfilerState {
#ifndef ANDROID
  TfLiteTelemetryProfilerStruct profilerStruct = {};
#endif
  // Key: event handle returned by ReportBeginOpInvokeEvent
  // Value: (op_name, op_idx, start_time)
  std::unordered_map<uint32_t,
                     std::tuple<std::string, int64_t, std::chrono::steady_clock::time_point>>
      pendingOps;
  std::vector<OpTiming> opTimings;
  std::atomic<uint32_t> nextHandle{0};

  void reset() {
    pendingOps.clear();
    opTimings.clear();
    nextHandle.store(0);
  }
};

#ifndef ANDROID

uint32_t TensorflowPlugin::profilerBeginOp(TfLiteTelemetryProfilerStruct* profiler,
                                           const char* op_name, int64_t op_idx,
                                           int64_t /* subgraph_idx */) {
  auto* state = static_cast<TensorflowProfilerState*>(profiler->data);
  uint32_t handle = state->nextHandle.fetch_add(1);
  state->pendingOps[handle] = {op_name ? std::string(op_name) : std::string(), op_idx,
                               std::chrono::steady_clock::now()};
  return handle;
}

void TensorflowPlugin::profilerEndOp(TfLiteTelemetryProfilerStruct* profiler,
                                     uint32_t event_handle) {
  auto endTime = std::chrono::steady_clock::now();
  auto* state = static_cast<TensorflowProfilerState*>(profiler->data);
  auto it = state->pendingOps.find(event_handle);
  if (it != state->pendingOps.end()) {
    auto& [name, opIdx, startTime] = it->second;
    double durationMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    state->opTimings.push_back({name, opIdx, durationMs});
    state->pendingOps.erase(it);
  }
}

void TensorflowPlugin::profilerOpEvent(TfLiteTelemetryProfilerStruct* profiler, const char* op_name,
                                       uint64_t elapsed_us, int64_t op_idx,
                                       int64_t /* subgraph_idx */) {
  // Some delegate ops report their own timing directly (elapsed_us in microseconds).
  auto* state = static_cast<TensorflowProfilerState*>(profiler->data);
  state->opTimings.push_back({op_name ? std::string(op_name) : std::string(), op_idx,
                              static_cast<double>(elapsed_us) / 1000.0});
}

// No-op stubs for required callbacks we don't use
static void profilerNoOpEvent(TfLiteTelemetryProfilerStruct*, const char*, uint64_t) {}
static void profilerNoOpTelemetry(TfLiteTelemetryProfilerStruct*, const char*, uint64_t) {}
static void profilerNoOpOpTelemetry(TfLiteTelemetryProfilerStruct*, const char*, int64_t, int64_t,
                                    uint64_t) {}
static void profilerNoOpSettings(TfLiteTelemetryProfilerStruct*, const char*,
                                 const TfLiteTelemetrySettings*) {}

#endif // !ANDROID

void TensorflowPlugin::installToRuntime(jsi::Runtime& runtime,
                                        std::shared_ptr<react::CallInvoker> callInvoker,
                                        FetchURLFunc fetchURL) {

  auto func = jsi::Function::createFromHostFunction(
      runtime, jsi::PropNameID::forAscii(runtime, "__loadTensorflowModel"), 1,
      [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
          size_t count) -> jsi::Value {
        auto start = std::chrono::steady_clock::now();
        auto modelPath = arguments[0].asString(runtime).utf8(runtime);

        log("Loading TensorFlow Lite Model from \"%s\"...", modelPath.c_str());

        Delegate delegateType = Delegate::Default;
        if (count > 1 && arguments[1].isString()) {
          auto delegate = arguments[1].asString(runtime).utf8(runtime);
          if (delegate == "core-ml") {
            delegateType = Delegate::CoreML;
          } else if (delegate == "metal") {
            delegateType = Delegate::Metal;
          } else if (delegate == "nnapi") {
            delegateType = Delegate::NnApi;
          } else if (delegate == "android-gpu") {
            delegateType = Delegate::AndroidGPU;
          } else {
            delegateType = Delegate::Default;
          }
        }

        int numThreads = 1;
        if (count > 2 && arguments[2].isNumber()) {
          numThreads = static_cast<int>(arguments[2].asNumber());
        }

        bool debugMode = false;
        if (count > 3 && arguments[3].isBool()) {
          debugMode = arguments[3].getBool();
        }

        // Run inference in fp16 (precision-loss allowed). For the CPU/default
        // delegate this applies XNNPACK FORCE_FP16; for GPU/Metal it toggles the
        // delegate's fp16 compute path.
        bool enableFp16 = false;
        if (count > 4 && arguments[4].isBool()) {
          enableFp16 = arguments[4].getBool();
        }

        auto promise = Promise::createPromise(runtime, [=, &runtime](
                                                           std::shared_ptr<Promise> promise) {
          // Launch async thread
          std::async(std::launch::async, [=, &runtime]() {
            try {
              // Fetch model from URL (JS bundle)
              Buffer buffer = fetchURL(modelPath);

              // Load Model into Tensorflow
              auto model = TfLiteModelCreate(buffer.data, buffer.size);
              if (model == nullptr) {
                callInvoker->invokeAsync(
                    [=]() { promise->reject("Failed to load model from \"" + modelPath + "\"!"); });
                return;
              }

              // Create TensorFlow Interpreter
              auto options = TfLiteInterpreterOptionsCreate();
              TfLiteInterpreterOptionsSetNumThreads(options, numThreads);

              switch (delegateType) {
#ifdef ANDROID
                case Delegate::CoreML: {
                  callInvoker->invokeAsync(
                      [=]() { promise->reject("CoreML Delegate is only supported on iOS!"); });
                  return;
                }
                case Delegate::Metal: {
                  callInvoker->invokeAsync(
                      [=]() { promise->reject("Metal Delegate is only supported on iOS!"); });
                  return;
                }
#else
                case Delegate::CoreML: {
                  if (!TFLIsCoreMLDelegateAvailable()) {
                    callInvoker->invokeAsync([=]() {
                      promise->reject("CoreML Delegate is not enabled! Set $EnableCoreMLDelegate to true in Podfile and rebuild.");
                    });
                    return;
                  }
                  auto delegate = TFLCreateCoreMLDelegate();
                  if (delegate != nullptr) {
                    TfLiteInterpreterOptionsAddDelegate(options, delegate);
                  }
                  break;
                }
                case Delegate::Metal: {
                  if (!TFLIsMetalDelegateAvailable()) {
                    callInvoker->invokeAsync([=]() {
                      promise->reject("Metal Delegate is not enabled! Set $EnableMetalDelegate to true in Podfile and rebuild.");
                    });
                    return;
                  }
                  auto delegate = TFLCreateMetalDelegate(enableFp16);
                  if (delegate != nullptr) {
                    TfLiteInterpreterOptionsAddDelegate(options, delegate);
                  }
                  break;
                }
#endif
#ifdef ANDROID
                case Delegate::NnApi: {
                  TfLiteNnapiDelegateOptions delegateOptions = TfLiteNnapiDelegateOptionsDefault();
                  auto delegate = TfLiteNnapiDelegateCreate(&delegateOptions);
                  TfLiteInterpreterOptionsAddDelegate(options, delegate);
                  break;
                }
                case Delegate::AndroidGPU: {
                  TfLiteGpuDelegateOptionsV2 delegateOptions = TfLiteGpuDelegateOptionsV2Default();
                  // Tuned for repeated, latency-sensitive inference (this lib runs
                  // the same model many times), not the default one-shot FP32 preset:
                  //  - FP16 compute (≈1.5-2x faster on mobile GPUs; minor precision loss)
                  //  - SUSTAINED_SPEED: keep the GPU warm across calls, allow better tuning
                  //  - MIN_LATENCY priority: optimize per-invoke latency
                  delegateOptions.is_precision_loss_allowed = enableFp16 ? 1 : 0;
                  delegateOptions.inference_preference =
                      TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
                  delegateOptions.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
                  auto delegate = TfLiteGpuDelegateV2Create(&delegateOptions);
                  TfLiteInterpreterOptionsAddDelegate(options, delegate);
                  break;
                }
#else
                case Delegate::NnApi: {
                  callInvoker->invokeAsync([=]() {
                    promise->reject("NNAPI Delegate is only supported on Android!");
                  });
                  return;
                }
                case Delegate::AndroidGPU: {
                  callInvoker->invokeAsync([=]() {
                    promise->reject("Android GPU Delegate is only supported on Android!");
                  });
                  return;
                }
#endif
                default: {
                  // Default CPU path. When fp16 is requested, attach an XNNPACK
                  // delegate forcing fp16 compute (fast on ARMv8.2-FP16 CPUs).
#ifdef ANDROID
                  if (enableFp16) {
                    FastTfliteXNNPackDelegateOptions xnnOptions =
                        TfLiteXNNPackDelegateOptionsDefault();
                    xnnOptions.num_threads = numThreads;
                    xnnOptions.flags |= FAST_TFLITE_XNNPACK_FLAG_FORCE_FP16;
                    auto xnn = TfLiteXNNPackDelegateCreate(&xnnOptions);
                    if (xnn != nullptr) {
                      TfLiteInterpreterOptionsAddDelegate(options, xnn);
                    }
                  }
#endif
                }
              }

              auto profilerState = std::make_shared<TensorflowProfilerState>();

              auto interpreter = TfLiteInterpreterCreate(model, options);

              if (interpreter == nullptr) {
                callInvoker->invokeAsync([=]() {
                  promise->reject("Failed to create TFLite interpreter from model \"" + modelPath +
                                  "\"!");
                });
                return;
              }

              // Initialize Model and allocate memory buffers
              auto plugin = std::make_shared<TensorflowPlugin>(
                  interpreter, buffer, delegateType, callInvoker, debugMode, profilerState);

              callInvoker->invokeAsync([=, &runtime]() {
                auto result = jsi::Object::createFromHostObject(runtime, plugin);
                promise->resolve(std::move(result));
              });

              auto end = std::chrono::steady_clock::now();
              log("Successfully loaded Tensorflow Model in %i ms!",
                  std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
            } catch (std::exception& error) {
              std::string message = error.what();
              callInvoker->invokeAsync([=]() { promise->reject(message); });
            }
          });
        });
        return promise;
      });

  runtime.global().setProperty(runtime, "__loadTensorflowModel", func);
}

std::string tfLiteStatusToString(TfLiteStatus status) {
  switch (status) {
    case kTfLiteOk:
      return "ok";
    case kTfLiteError:
      return "error";
    case kTfLiteDelegateError:
      return "delegate-error";
    case kTfLiteApplicationError:
      return "application-error";
    case kTfLiteDelegateDataNotFound:
      return "delegate-data-not-found";
    case kTfLiteDelegateDataWriteError:
      return "delegate-data-write-error";
    case kTfLiteDelegateDataReadError:
      return "delegate-data-read-error";
    case kTfLiteUnresolvedOps:
      return "unresolved-ops";
    case kTfLiteCancelled:
      return "cancelled";
    default:
      return "unknown";
  }
}

TensorflowPlugin::TensorflowPlugin(TfLiteInterpreter* interpreter, Buffer model, Delegate delegate,
                                   std::shared_ptr<react::CallInvoker> callInvoker, bool debugMode,
                                   std::shared_ptr<TensorflowProfilerState> profilerState)
    : _interpreter(interpreter), _delegate(delegate), _model(model), _callInvoker(callInvoker),
      _debugMode(debugMode), _profilerState(std::move(profilerState)) {
  // Allocate memory for the model's input/output `TFLTensor`s.
  TfLiteStatus status = TfLiteInterpreterAllocateTensors(_interpreter);
  if (status != kTfLiteOk) {
    [[unlikely]];
    throw std::runtime_error(
        "TFLite: Failed to allocate memory for input/output tensors! Status: " +
        tfLiteStatusToString(status));
  }

  log("Successfully created Tensorflow Plugin! (debug mode: %s)", debugMode ? "on" : "off");
}

TensorflowPlugin::~TensorflowPlugin() {
  if (_model.data != nullptr) {
    free(_model.data);
    _model.data = nullptr;
    _model.size = 0;
  }
  if (_interpreter != nullptr) {
    TfLiteInterpreterDelete(_interpreter);
    _interpreter = nullptr;
  }
}

std::string TensorflowPlugin::getDelegateHardwareName() const {
  switch (_delegate) {
    case Delegate::Metal:
      return "metal";
    case Delegate::CoreML:
      return "core-ml";
    case Delegate::NnApi:
      return "nnapi";
    case Delegate::AndroidGPU:
      return "android-gpu";
    default:
      return "cpu";
  }
}

std::vector<TensorDebugInfo> TensorflowPlugin::inspectIOTensors() const {
  std::vector<TensorDebugInfo> tensors;
  std::string delegateHw = getDelegateHardwareName();

  // Helper: collect one tensor
  auto collectTensor = [&](const TfLiteTensor* tensor, int reportedIndex) {
    if (tensor == nullptr)
      return;

    TfLiteAllocationType allocType = tensor->allocation_type;
    // Skip constant / weight tensors — they aren't compute tensors.
    if (allocType == kTfLiteMmapRo || allocType == kTfLiteMemNone ||
        allocType == kTfLitePersistentRo) {
      return;
    }

    TensorDebugInfo info;
    info.index = reportedIndex;
    info.name = (tensor->name != nullptr) ? std::string(tensor->name)
                                          : ("#" + std::to_string(reportedIndex));

    if (tensor->dims != nullptr) {
      for (int d = 0; d < tensor->dims->size; d++) {
        info.shape.push_back(tensor->dims->data[d]);
      }
    }

    // A tensor was processed by a hardware delegate when:
    //   • Its delegate pointer is set (delegate owns the buffer), OR
    //   • It has a valid delegate buffer handle (buffer lives in delegate memory), OR
    //   • Its allocation type is kTfLiteNonCpu (AHWB / GPU memory — Android LiteRt only).
    bool isDelegateManaged =
        (tensor->delegate != nullptr) || (tensor->buffer_handle != kTfLiteNullBufferHandle);
#ifdef ANDROID
    isDelegateManaged = isDelegateManaged || (allocType == kTfLiteNonCpu);
#endif
    info.hardware = isDelegateManaged ? delegateHw : "cpu";

    tensors.push_back(std::move(info));
  };

  // Use only the stable C API — iterate input and output tensors.
  // The experimental TfLiteInterpreterGetOutputTensorIndex API is not used
  // because it can return undefined values on some platforms/versions and
  // trigger an effectively infinite loop.
  int inputCount = TfLiteInterpreterGetInputTensorCount(_interpreter);
  for (int i = 0; i < inputCount; i++) {
    collectTensor(TfLiteInterpreterGetInputTensor(_interpreter, i), i);
  }

  int outputCount = TfLiteInterpreterGetOutputTensorCount(_interpreter);
  for (int i = 0; i < outputCount; i++) {
    collectTensor(TfLiteInterpreterGetOutputTensor(_interpreter, i), inputCount + i);
  }

  return tensors;
}

InferenceStats TensorflowPlugin::collectDebugStats(double totalMs) {
  InferenceStats stats;
  stats.totalTimeMs = totalMs;
  stats.tensors = inspectIOTensors();

  // Copy per-op timings recorded by the telemetry profiler callbacks
  if (_profilerState && !_profilerState->opTimings.empty()) {
    stats.opTimings = _profilerState->opTimings;
  }

  return stats;
}

void TensorflowPlugin::printDebugStats(const InferenceStats& stats) {
  std::string delegateHw = getDelegateHardwareName();

  // Count tensors by hardware
  int cpuCount = 0, delegateCount = 0;
  for (const auto& t : stats.tensors) {
    if (t.hardware == "cpu")
      cpuCount++;
    else
      delegateCount++;
  }

  log("[TFLite] Inference: %.3f ms | delegate: %s | tensors: %d cpu, %d %s", stats.totalTimeMs,
      delegateHw.c_str(), cpuCount, delegateCount, delegateHw.c_str());

  for (const auto& t : stats.tensors) {
    // Build shape string
    std::ostringstream shapeStream;
    shapeStream << "[";
    for (size_t j = 0; j < t.shape.size(); j++) {
      if (j > 0)
        shapeStream << ",";
      shapeStream << t.shape[j];
    }
    shapeStream << "]";

    bool isCpuFallback = (t.hardware == "cpu") && (_delegate != Delegate::Default);
    log("[TFLite]   [%d] '%s' %s → %s%s", t.index, t.name.c_str(), shapeStream.str().c_str(),
        t.hardware.c_str(), isCpuFallback ? " ⚠ CPU fallback" : "");
  }
}

std::shared_ptr<TypedArrayBase>
TensorflowPlugin::getOutputArrayForTensor(jsi::Runtime& runtime, const TfLiteTensor* tensor) {
  auto name = std::string(TfLiteTensorName(tensor));
  if (_outputBuffers.find(name) == _outputBuffers.end()) {
    _outputBuffers[name] =
        std::make_shared<TypedArrayBase>(TensorHelpers::createJSBufferForTensor(runtime, tensor));
  }
  return _outputBuffers[name];
}

void TensorflowPlugin::copyInputBuffers(jsi::Runtime& runtime, jsi::Object inputValues) {
  // Input has to be array in input tensor size
#if DEBUG
  if (!inputValues.isArray(runtime)) {
    [[unlikely]];
    throw jsi::JSError(runtime,
                       "TFLite: Input Values must be an array, one item for each input tensor!");
  }
#endif

  jsi::Array array = inputValues.asArray(runtime);
  size_t count = array.size(runtime);
  if (count != TfLiteInterpreterGetInputTensorCount(_interpreter)) {
    [[unlikely]];
    throw jsi::JSError(runtime,
                       "TFLite: Input Values have different size than there are input tensors!");
  }

  for (size_t i = 0; i < count; i++) {
    TfLiteTensor* tensor = TfLiteInterpreterGetInputTensor(_interpreter, i);
    jsi::Object object = array.getValueAtIndex(runtime, i).asObject(runtime);

#if DEBUG
    if (!isTypedArray(runtime, object)) {
      [[unlikely]];
      throw jsi::JSError(
          runtime,
          "TFLite: Input value is not a TypedArray! (Uint8Array, Uint16Array, Float32Array, etc.)");
    }
#endif

    TypedArrayBase inputBuffer = getTypedArray(runtime, std::move(object));
    TensorHelpers::updateTensorFromJSBuffer(runtime, tensor, inputBuffer);
  }
}

jsi::Value TensorflowPlugin::copyOutputBuffers(jsi::Runtime& runtime) {
  // Copy output to result process the inference results.
  int outputTensorsCount = TfLiteInterpreterGetOutputTensorCount(_interpreter);
  jsi::Array result(runtime, outputTensorsCount);
  for (size_t i = 0; i < outputTensorsCount; i++) {
    const TfLiteTensor* outputTensor = TfLiteInterpreterGetOutputTensor(_interpreter, i);
    auto outputBuffer = getOutputArrayForTensor(runtime, outputTensor);
    TensorHelpers::updateJSBufferFromTensor(runtime, *outputBuffer, outputTensor);
    result.setValueAtIndex(runtime, i, *outputBuffer);
  }
  return result;
}

void TensorflowPlugin::run() {
  if (_debugMode) {
    // Reset per-op timing log so each inference gets a fresh slate
    if (_profilerState)
      _profilerState->reset();

    auto startTime = std::chrono::steady_clock::now();
    TfLiteStatus status = TfLiteInterpreterInvoke(_interpreter);
    auto endTime = std::chrono::steady_clock::now();

    double totalMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    _lastStats = collectDebugStats(totalMs);
    _hasStats = true;
    // NOTE: Do NOT call printDebugStats here. On iOS, printf/vprintf write to
    // stdout which may be a pipe that blocks when its buffer fills (e.g. on a
    // physical device not attached to Xcode). All logging is done on the JS
    // thread via console.log inside the JS wrapWithDebugLogging wrapper.

    if (status != kTfLiteOk) {
      [[unlikely]];
      throw std::runtime_error("TFLite: Failed to run TFLite Model! Status: " +
                               tfLiteStatusToString(status));
    }
  } else {
    TfLiteStatus status = TfLiteInterpreterInvoke(_interpreter);
    if (status != kTfLiteOk) {
      [[unlikely]];
      throw std::runtime_error("TFLite: Failed to run TFLite Model! Status: " +
                               tfLiteStatusToString(status));
    }
  }
}

jsi::Value TensorflowPlugin::get(jsi::Runtime& runtime, const jsi::PropNameID& propNameId) {
  auto propName = propNameId.utf8(runtime);

  if (propName == "runSync") {
    return jsi::Function::createFromHostFunction(
        runtime, jsi::PropNameID::forAscii(runtime, "runModel"), 1,
        [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
            size_t count) -> jsi::Value {
          // 1.
          copyInputBuffers(runtime, arguments[0].asObject(runtime));
          // 2.
          this->run();
          // 3.
          return copyOutputBuffers(runtime);
        });
  } else if (propName == "run") {
    return jsi::Function::createFromHostFunction(
        runtime, jsi::PropNameID::forAscii(runtime, "runModel"), 1,
        [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
            size_t count) -> jsi::Value {
          // 1.
          copyInputBuffers(runtime, arguments[0].asObject(runtime));
          auto promise =
              Promise::createPromise(runtime, [=, &runtime](std::shared_ptr<Promise> promise) {
                std::async(std::launch::async, [=, &runtime]() {
                  // 2.
                  try {
                    this->run();

                    this->_callInvoker->invokeAsync([=, &runtime]() {
                      // 3.
                      auto result = this->copyOutputBuffers(runtime);
                      promise->resolve(std::move(result));
                    });
                  } catch (std::exception& error) {
                    promise->reject(error.what());
                  }
                });
              });
          return promise;
        });
  } else if (propName == "inputs") {
    int size = TfLiteInterpreterGetInputTensorCount(_interpreter);
    jsi::Array tensors(runtime, size);
    for (size_t i = 0; i < size; i++) {
      TfLiteTensor* tensor = TfLiteInterpreterGetInputTensor(_interpreter, i);
      if (tensor == nullptr) {
        [[unlikely]];
        throw jsi::JSError(runtime,
                           "TFLite: Failed to get input tensor " + std::to_string(i) + "!");
      }

      jsi::Object object = TensorHelpers::tensorToJSObject(runtime, tensor);
      tensors.setValueAtIndex(runtime, i, object);
    }
    return tensors;
  } else if (propName == "outputs") {
    int size = TfLiteInterpreterGetOutputTensorCount(_interpreter);
    jsi::Array tensors(runtime, size);
    for (size_t i = 0; i < size; i++) {
      const TfLiteTensor* tensor = TfLiteInterpreterGetOutputTensor(_interpreter, i);
      if (tensor == nullptr) {
        [[unlikely]];
        throw jsi::JSError(runtime,
                           "TFLite: Failed to get output tensor " + std::to_string(i) + "!");
      }

      jsi::Object object = TensorHelpers::tensorToJSObject(runtime, tensor);
      tensors.setValueAtIndex(runtime, i, object);
    }
    return tensors;
  } else if (propName == "delegate") {
    switch (_delegate) {
      case Delegate::Default:
        return jsi::String::createFromUtf8(runtime, "default");
      case Delegate::CoreML:
        return jsi::String::createFromUtf8(runtime, "core-ml");
      case Delegate::Metal:
        return jsi::String::createFromUtf8(runtime, "metal");
      case Delegate::NnApi:
        return jsi::String::createFromUtf8(runtime, "nnapi");
      case Delegate::AndroidGPU:
        return jsi::String::createFromUtf8(runtime, "android-gpu");
    }
  } else if (propName == "stats") {
    if (!_debugMode || !_hasStats) {
      return jsi::Value::undefined();
    }

    jsi::Object statsObj(runtime);
    statsObj.setProperty(runtime, "totalTimeMs", _lastStats.totalTimeMs);

    jsi::Array tensorsArr(runtime, _lastStats.tensors.size());
    for (size_t i = 0; i < _lastStats.tensors.size(); i++) {
      const auto& t = _lastStats.tensors[i];
      jsi::Object tensorObj(runtime);
      tensorObj.setProperty(runtime, "index", t.index);
      tensorObj.setProperty(runtime, "name", jsi::String::createFromUtf8(runtime, t.name));
      tensorObj.setProperty(runtime, "hardware", jsi::String::createFromUtf8(runtime, t.hardware));

      jsi::Array shapeArr(runtime, t.shape.size());
      for (size_t j = 0; j < t.shape.size(); j++) {
        shapeArr.setValueAtIndex(runtime, j, t.shape[j]);
      }
      tensorObj.setProperty(runtime, "shape", std::move(shapeArr));

      tensorsArr.setValueAtIndex(runtime, i, std::move(tensorObj));
    }
    statsObj.setProperty(runtime, "tensors", std::move(tensorsArr));

    // Per-op timings (populated on iOS; empty array on Android)
    jsi::Array opTimingsArr(runtime, _lastStats.opTimings.size());
    for (size_t i = 0; i < _lastStats.opTimings.size(); i++) {
      const auto& op = _lastStats.opTimings[i];
      jsi::Object opObj(runtime);
      opObj.setProperty(runtime, "name", jsi::String::createFromUtf8(runtime, op.name));
      opObj.setProperty(runtime, "opIdx", static_cast<double>(op.opIdx));
      opObj.setProperty(runtime, "durationMs", op.durationMs);
      opTimingsArr.setValueAtIndex(runtime, i, std::move(opObj));
    }
    statsObj.setProperty(runtime, "opTimings", std::move(opTimingsArr));

    return statsObj;
  }

  return jsi::HostObject::get(runtime, propNameId);
}

std::vector<jsi::PropNameID> TensorflowPlugin::getPropertyNames(jsi::Runtime& runtime) {
  std::vector<jsi::PropNameID> result;
  result.push_back(jsi::PropNameID::forAscii(runtime, "run"));
  result.push_back(jsi::PropNameID::forAscii(runtime, "runSync"));
  result.push_back(jsi::PropNameID::forAscii(runtime, "inputs"));
  result.push_back(jsi::PropNameID::forAscii(runtime, "outputs"));
  result.push_back(jsi::PropNameID::forAscii(runtime, "delegate"));
  result.push_back(jsi::PropNameID::forAscii(runtime, "stats"));
  return result;
}
