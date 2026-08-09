//
//  JumpProcessor.mm
//  react-native-fast-tflite
//
//  Implementation of the native ring + background-thread full-model runner.
//  Stored as .mm so the same source compiles on iOS (Objective-C++) and, via
//  the CMake `-x c++` override, as plain C++ on Android. No ObjC is used here.
//

#include "JumpProcessor.h"

#include "TensorHelpers.h"
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <utility>

using namespace facebook;
using namespace mrousavy;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int readIntProp(jsi::Runtime& runtime, const jsi::Object& obj, const char* name) {
  jsi::Value v = obj.getProperty(runtime, name);
  if (!v.isNumber()) {
    throw jsi::JSError(runtime, std::string("JumpProcessor: config.") + name + " must be a number");
  }
  return static_cast<int>(v.asNumber());
}

/**
 * Split a slot-major [1, N, C] signal head into its feet and rope channels.
 *
 * The models emit every per-slot head as `value[slot * C + channel]`; C is
 * inferred from the tensor's float count so a head that gains channels needs no
 * native change. `feet` and `rope` are each filled with exactly `bufferSize`
 * values; a channel the tensor doesn't have is left empty (a head with C == 1
 * yields feet only). Returns false when the tensor is unusable, leaving both
 * outputs empty.
 */
static bool readSignalChannels(const TfLiteTensor* tensor, int bufferSize, std::vector<float>& feet,
                               std::vector<float>& rope) {
  feet.clear();
  rope.clear();
  if (tensor == nullptr || bufferSize <= 0)
    return false;

  const size_t byteSize = TfLiteTensorByteSize(tensor);
  const size_t floatCount = byteSize / sizeof(float);
  const int channels = static_cast<int>(floatCount / static_cast<size_t>(bufferSize));
  if (channels <= 0)
    return false;

  std::vector<float> raw(floatCount);
  if (TfLiteTensorCopyToBuffer(tensor, raw.data(), byteSize) != kTfLiteOk)
    return false;

  const auto extract = [&](int channel, std::vector<float>& out) {
    if (channel >= channels)
      return;
    out.assign(bufferSize, 0.0f);
    for (int i = 0; i < bufferSize; i++) {
      const size_t idx = static_cast<size_t>(i) * channels + channel;
      if (idx >= floatCount)
        break;
      out[i] = raw[idx];
    }
  };
  extract(kJumpChannelFeet, feet);
  extract(kJumpChannelRope, rope);
  return true;
}

// ---------------------------------------------------------------------------
// Construction / teardown
// ---------------------------------------------------------------------------

JumpProcessor::JumpProcessor(std::shared_ptr<TensorflowPlugin> fullModel,
                             JumpProcessorConfig config,
                             std::shared_ptr<react::CallInvoker> callInvoker,
                             std::shared_ptr<jsi::Function> onResult, jsi::Runtime& runtime)
    : _fullModel(std::move(fullModel)), _config(config), _callInvoker(std::move(callInvoker)),
      _onResult(std::move(onResult)), _runtime(runtime) {
  _interpreter = _fullModel->getInterpreter();
  if (_interpreter == nullptr) {
    throw std::runtime_error("JumpProcessor: full model has no interpreter");
  }
  if (_config.bufferSize <= 0 || _config.outputSize <= 0 || _config.fullModelInterval <= 0) {
    throw std::runtime_error("JumpProcessor: invalid config (bufferSize/outputSize/"
                             "fullModelInterval must be > 0)");
  }

  _ring.assign(static_cast<size_t>(_config.bufferSize) * _config.outputSize, 0.0f);
  _snapshot.assign(static_cast<size_t>(_config.bufferSize) * _config.outputSize, 0.0f);

  _fullModelInterval.store(_config.fullModelInterval);

  _worker = std::thread(&JumpProcessor::workerLoop, this);
}

JumpProcessor::~JumpProcessor() {
  dispose();
}

void JumpProcessor::dispose() {
  bool wasStopped = _stop.exchange(true);
  if (wasStopped)
    return; // idempotent

  {
    std::lock_guard<std::mutex> lock(_workMutex);
    _workPending = true; // wake the worker so it can observe _stop
  }
  _workCv.notify_all();
  if (_worker.joinable()) {
    _worker.join();
  }
}

// ---------------------------------------------------------------------------
// Frame ingestion (called from the camera worklet thread)
// ---------------------------------------------------------------------------

void JumpProcessor::pushBackbone(const float* data, size_t length, int64_t /*frameNumber*/) {
  const size_t outputSize = static_cast<size_t>(_config.outputSize);
  const int bufferSize = _config.bufferSize;
  if (length < outputSize) {
    // Not enough data — ignore this frame rather than read out of bounds.
    return;
  }

  int64_t totalFrames;
  // Whether this push completes a full-model interval and no run is in flight.
  bool willTrigger = false;
  {
    std::lock_guard<std::mutex> lock(_ringMutex);
    const int writeIdx = _writeIndex;
    std::memcpy(_ring.data() + static_cast<size_t>(writeIdx) * outputSize, data,
                outputSize * sizeof(float));
    _writeIndex = (writeIdx + 1) % bufferSize;
    _frameCount += 1;
    totalFrames = _frameCount;

    // Decide whether to trigger a run, and if so, snapshot the window NOW —
    // synchronously, on the camera thread, while still holding the ring lock.
    // This is the critical fix: gathering on the worker thread later would let
    // subsequent pushBackbone calls overwrite the oldest slots of this window
    // before the worker reads them, splicing newer frames into the start of the
    // sequence and destroying temporal continuity (periodicity collapses). The
    // old single-thread pipeline gathered synchronously for exactly this reason.
    //
    // After this push, the ring in temporal order [oldest..newest] starts at the
    // new _writeIndex (the slot we'll overwrite next == the oldest frame).
    const int interval = _fullModelInterval.load();
    if (totalFrames >= bufferSize && interval > 0 && totalFrames % interval == 0) {
      bool expected = false;
      if (_busy.compare_exchange_strong(expected, true)) {
        willTrigger = true;
        const int oldestIdx = _writeIndex; // == (writeIdx + 1) % bufferSize
        const auto gStart = std::chrono::steady_clock::now();
        for (int i = 0; i < bufferSize; i++) {
          const int bufIdx = (oldestIdx + i) % bufferSize;
          std::memcpy(_snapshot.data() + static_cast<size_t>(i) * outputSize,
                      _ring.data() + static_cast<size_t>(bufIdx) * outputSize,
                      outputSize * sizeof(float));
        }
        const auto gEnd = std::chrono::steady_clock::now();
        _lastGatherMs.store(std::chrono::duration<double, std::milli>(gEnd - gStart).count());
        _pendingWriteIdx = oldestIdx;
        // Absolute global frame index of the oldest frame now in the window.
        // After this push the window holds frames [totalFrames-bufferSize ..
        // totalFrames-1]; slot 0 (oldest) is the first of that range.
        _pendingOldestFrame = totalFrames - bufferSize;
        _pendingGeneration = _generation.load();
      }
      // If busy, this interval is skipped; the next interval will catch up.
    }
  }
  _framesPushed.fetch_add(1);

  // Signal the worker (snapshot already taken under the lock above).
  if (willTrigger) {
    _runsTriggered.fetch_add(1);
    {
      std::lock_guard<std::mutex> lock(_workMutex);
      _workPending = true;
    }
    _workCv.notify_one();
  }
}

void JumpProcessor::reset() {
  // Bump generation so any in-flight result is discarded on delivery.
  _generation.fetch_add(1);
  std::lock_guard<std::mutex> lock(_ringMutex);
  std::fill(_ring.begin(), _ring.end(), 0.0f);
  _writeIndex = 0;
  _frameCount = 0;
}

void JumpProcessor::setFullModelInterval(int interval) {
  // Clamp to >= 1. Lock-free: pushBackbone reads this atomically on the next
  // frame, so the new cadence takes effect immediately with no ring reset and
  // no in-flight run disruption.
  if (interval < 1)
    interval = 1;
  _fullModelInterval.store(interval);
}

// ---------------------------------------------------------------------------
// Worker thread
// ---------------------------------------------------------------------------

void JumpProcessor::workerLoop() {
  while (true) {
    int writeIdx;
    int64_t oldestFrame;
    uint64_t generation;
    {
      std::unique_lock<std::mutex> lock(_workMutex);
      _workCv.wait(lock, [this] { return _workPending || _stop.load(); });
      if (_stop.load())
        return;
      _workPending = false;
      writeIdx = _pendingWriteIdx;
      oldestFrame = _pendingOldestFrame;
      generation = _pendingGeneration;
    }

    runFullModelOnWorker(generation, writeIdx, oldestFrame);
    _busy.store(false);
  }
}

void JumpProcessor::runFullModelOnWorker(uint64_t generation, int writeIdx,
                                         int64_t oldestFrameNumber) {
  // Skip-reason codes mirror the _lastSkip doc in the header.
  enum SkipReason {
    Ok = 0,
    ResetGen = 1,
    InputNull = 2,
    CopyFailed = 3,
    InvokeFailed = 4,
    OutputNull = 5
  };
  _runsStarted.fetch_add(1);

  const int bufferSize = _config.bufferSize;

  // The window was already snapshotted into _snapshot synchronously in
  // pushBackbone (under the ring lock, at trigger time), so it is a consistent
  // oldest→newest sequence frozen at the trigger. _busy guarantees no other run
  // (and thus no other snapshot write) is in flight, so _snapshot is stable here.
  // gatherMs is therefore ~0 here (the gather cost now lives on the camera
  // thread); we report it as 0 to keep the stats shape unchanged.
  const double gatherMsZero = 0.0;

  // Bail if a reset happened since the trigger.
  if (generation != _generation.load()) {
    _lastSkip.store(ResetGen);
    return;
  }

  auto t1 = std::chrono::steady_clock::now();

  // Feed the snapshot into the full model's single input tensor and invoke.
  TfLiteTensor* inputTensor = TfLiteInterpreterGetInputTensor(_interpreter, 0);
  if (inputTensor == nullptr) {
    _lastSkip.store(InputNull);
    return;
  }
  const size_t snapshotBytes = _snapshot.size() * sizeof(float);
  const size_t tensorBytes = TfLiteTensorByteSize(inputTensor);
  _expectedBytes.store(static_cast<int>(snapshotBytes));
  _lastInputBytes.store(static_cast<int>(tensorBytes));
  // Copy as many bytes as both sides can hold. The old (working) path copied the
  // JS buffer's byte length directly with only a DEBUG-time exact-size assert, so
  // we mirror that tolerance instead of hard-failing on any mismatch.
  const size_t copyBytes = std::min(snapshotBytes, tensorBytes);
  const auto tCopyStart = std::chrono::steady_clock::now();
  TfLiteStatus copyStatus = TfLiteTensorCopyFromBuffer(inputTensor, _snapshot.data(), copyBytes);
  if (copyStatus != kTfLiteOk) {
    _lastSkip.store(CopyFailed);
    return;
  }
  const auto tInvokeStart = std::chrono::steady_clock::now();

  TfLiteStatus status = TfLiteInterpreterInvoke(_interpreter);
  if (status != kTfLiteOk) {
    _lastSkip.store(InvokeFailed);
    return;
  }

  auto t2 = std::chrono::steady_clock::now();
  // Sub-step breakdown: input copy vs the actual GPU/CPU compute.
  _lastCopyInMs.store(std::chrono::duration<double, std::milli>(tInvokeStart - tCopyStart).count());
  _lastInvokeMs.store(std::chrono::duration<double, std::milli>(t2 - tInvokeStart).count());

  // Extract periodicities and periods from the configured output tensors.
  const TfLiteTensor* periodicityTensor =
      TfLiteInterpreterGetOutputTensor(_interpreter, _config.outputTensorPeriodicity);
  const TfLiteTensor* periodTensor =
      TfLiteInterpreterGetOutputTensor(_interpreter, _config.outputTensorPeriod);
  if (periodicityTensor == nullptr || periodTensor == nullptr) {
    _lastSkip.store(OutputNull);
    return;
  }
  _lastSkip.store(Ok);

  // Capture I/O tensor hardware placement once (it's stable across runs). Lets
  // the benchmark screen show whether the model's tensors are on the GPU
  // delegate or fell back to the CPU. Done here on the worker thread after a
  // successful invoke so the interpreter isn't being read concurrently.
  if (!_hasPlacement.load()) {
    auto io = _fullModel->inspectIOTensors();
    {
      std::lock_guard<std::mutex> lock(_placementMutex);
      _placement = std::move(io);
    }
    _hasPlacement.store(true);
  }

  // Every signal head is slot-major [1, N, C] with the feet channel first and
  // the rope channel second (see readSignalChannels). Both channels are
  // delivered: the feet arrays drive counting today, the rope arrays let JS pick
  // up the rope signal without another native rebuild.
  auto periodicities = std::make_shared<std::vector<float>>(bufferSize, 0.0f);
  auto periods = std::make_shared<std::vector<float>>(bufferSize, 0.0f);
  auto periodicitiesRope = std::make_shared<std::vector<float>>();
  auto periodsRope = std::make_shared<std::vector<float>>();
  readSignalChannels(periodicityTensor, bufferSize, *periodicities, *periodicitiesRope);
  readSignalChannels(periodTensor, bufferSize, *periods, *periodsRope);
  // readSignalChannels leaves feet empty on an unusable tensor; keep the
  // delivered length at bufferSize so JS never sees a short array.
  if (periodicities->empty())
    periodicities->assign(bufferSize, 0.0f);
  if (periods->empty())
    periods->assign(bufferSize, 0.0f);

  // Optional marks tensor. Empty when not configured (-1); JS then falls back to
  // period-only counting.
  auto marks = std::make_shared<std::vector<float>>();
  auto marksRope = std::make_shared<std::vector<float>>();
  if (_config.outputTensorMarks >= 0) {
    const TfLiteTensor* marksTensor =
        TfLiteInterpreterGetOutputTensor(_interpreter, _config.outputTensorMarks);
    readSignalChannels(marksTensor, bufferSize, *marks, *marksRope);
  }

  // Optional event-type tensor ([1,N,C], C class logits per slot). Empty when
  // not configured (-1). We take the per-slot argmax here (native) and deliver
  // one class index per slot — the raw logits never cross to JS. C is inferred
  // from the tensor size / bufferSize; layout is slot-major (logits[i*C + c]),
  // same as the signal heads (see readSignalChannels) — here every channel is
  // read, not just one.
  auto eventTypes = std::make_shared<std::vector<float>>();
  if (_config.outputTensorEventType >= 0) {
    const TfLiteTensor* eventTypeTensor =
        TfLiteInterpreterGetOutputTensor(_interpreter, _config.outputTensorEventType);
    if (eventTypeTensor != nullptr && bufferSize > 0) {
      eventTypes->assign(bufferSize, 0.0f);
      const size_t eByteSize = TfLiteTensorByteSize(eventTypeTensor);
      std::vector<float> eRaw(eByteSize / sizeof(float));
      TfLiteTensorCopyToBuffer(eventTypeTensor, eRaw.data(), eByteSize);
      const int numClasses = static_cast<int>(eRaw.size()) / bufferSize;
      if (numClasses > 0) {
        for (int i = 0; i < bufferSize; i++) {
          const size_t base = static_cast<size_t>(i) * numClasses;
          if (base + numClasses > eRaw.size())
            break;
          int argmax = 0;
          float best = eRaw[base];
          for (int c = 1; c < numClasses; c++) {
            const float v = eRaw[base + c];
            if (v > best) {
              best = v;
              argmax = c;
            }
          }
          (*eventTypes)[i] = static_cast<float>(argmax);
        }
      }
    }
  }

  const auto tCopyOutEnd = std::chrono::steady_clock::now();
  _lastCopyOutMs.store(std::chrono::duration<double, std::milli>(tCopyOutEnd - t2).count());

  const double gatherMs = gatherMsZero;
  const double inferenceMs = std::chrono::duration<double, std::milli>(t2 - t1).count();

  // Deliver the small result to JS. Capture by value (shared_ptrs) so the data
  // outlives this worker frame. Re-check generation on the JS thread too.
  auto onResult = _onResult;
  auto callInvoker = _callInvoker;
  auto* self = this;
  callInvoker->invokeAsync([self, onResult, periodicities, periods, marks, eventTypes,
                            periodicitiesRope, periodsRope, marksRope, writeIdx, oldestFrameNumber,
                            gatherMs, inferenceMs, generation]() {
    if (self->_stop.load())
      return;
    if (generation != self->_generation.load())
      return;
    jsi::Runtime& rt = self->_runtime;

    // Build Float32Arrays from the vectors. Use the vector constructor (it
    // copies with element semantics); do NOT use updateUnsafe(data, count) —
    // that helper treats its length arg as BYTES, so passing an element count
    // throws a size-mismatch and the result silently never reaches JS.
    auto toFloat32Array = [&](const std::vector<float>& src) {
      return TypedArray<TypedArrayKind::Float32Array>(rt, src);
    };

    try {
      onResult->call(rt, toFloat32Array(*periodicities), toFloat32Array(*periods),
                     jsi::Value(writeIdx), jsi::Value(gatherMs), jsi::Value(inferenceMs),
                     jsi::Value(static_cast<double>(oldestFrameNumber)), toFloat32Array(*marks),
                     toFloat32Array(*eventTypes), toFloat32Array(*periodicitiesRope),
                     toFloat32Array(*periodsRope), toFloat32Array(*marksRope));
      self->_runsCompleted.fetch_add(1);
    } catch (jsi::JSError& e) {
      // A JS callback error must not crash the app. Record it so the failure
      // is visible in stats rather than silently swallowed (this is exactly
      // how the earlier byte/element bug hid itself).
      self->_lastSkip.store(6); // 6 = deliver-threw (see header doc)
    }
  });
}

// ---------------------------------------------------------------------------
// HostObject surface
// ---------------------------------------------------------------------------

jsi::Value JumpProcessor::get(jsi::Runtime& runtime, const jsi::PropNameID& propNameId) {
  auto propName = propNameId.utf8(runtime);

  if (propName == "pushBackbone") {
    return jsi::Function::createFromHostFunction(
        runtime, jsi::PropNameID::forAscii(runtime, "pushBackbone"), 2,
        [this](jsi::Runtime& rt, const jsi::Value&, const jsi::Value* args,
               size_t count) -> jsi::Value {
          if (count < 1 || !args[0].isObject())
            return jsi::Value::undefined();
          jsi::Object obj = args[0].asObject(rt);
          if (!isTypedArray(rt, obj))
            return jsi::Value::undefined();
          auto typed = getTypedArray(rt, std::move(obj)).get<TypedArrayKind::Float32Array>(rt);
          auto vec = typed.toVector(rt);
          int64_t frameNumber =
              count > 1 && args[1].isNumber() ? static_cast<int64_t>(args[1].asNumber()) : 0;
          this->pushBackbone(vec.data(), vec.size(), frameNumber);
          return jsi::Value::undefined();
        });
  } else if (propName == "setFullModelInterval") {
    return jsi::Function::createFromHostFunction(
        runtime, jsi::PropNameID::forAscii(runtime, "setFullModelInterval"), 1,
        [this](jsi::Runtime& rt, const jsi::Value&, const jsi::Value* args,
               size_t count) -> jsi::Value {
          if (count < 1 || !args[0].isNumber())
            return jsi::Value::undefined();
          this->setFullModelInterval(static_cast<int>(args[0].asNumber()));
          return jsi::Value::undefined();
        });
  } else if (propName == "reset") {
    return jsi::Function::createFromHostFunction(
        runtime, jsi::PropNameID::forAscii(runtime, "reset"), 0,
        [this](jsi::Runtime&, const jsi::Value&, const jsi::Value*, size_t) -> jsi::Value {
          this->reset();
          return jsi::Value::undefined();
        });
  } else if (propName == "dispose") {
    return jsi::Function::createFromHostFunction(
        runtime, jsi::PropNameID::forAscii(runtime, "dispose"), 0,
        [this](jsi::Runtime&, const jsi::Value&, const jsi::Value*, size_t) -> jsi::Value {
          this->dispose();
          return jsi::Value::undefined();
        });
  } else if (propName == "stats") {
    // Debug observability: snapshot of the internal counters. Lets the dev
    // screen see exactly where the pipeline stalls (e.g. runs triggered but not
    // completed → a worker early-return, with lastSkip naming which one).
    jsi::Object obj(runtime);
    obj.setProperty(runtime, "framesPushed", static_cast<double>(_framesPushed.load()));
    obj.setProperty(runtime, "runsTriggered", static_cast<double>(_runsTriggered.load()));
    obj.setProperty(runtime, "runsStarted", static_cast<double>(_runsStarted.load()));
    obj.setProperty(runtime, "runsCompleted", static_cast<double>(_runsCompleted.load()));
    obj.setProperty(runtime, "lastSkip", static_cast<double>(_lastSkip.load()));
    obj.setProperty(runtime, "inputTensorBytes", static_cast<double>(_lastInputBytes.load()));
    obj.setProperty(runtime, "snapshotBytes", static_cast<double>(_expectedBytes.load()));
    obj.setProperty(runtime, "busy", _busy.load());

    // Benchmark sub-step timings of the most recent full-model run (ms).
    obj.setProperty(runtime, "gatherMs", _lastGatherMs.load());
    obj.setProperty(runtime, "copyInMs", _lastCopyInMs.load());
    obj.setProperty(runtime, "invokeMs", _lastInvokeMs.load());
    obj.setProperty(runtime, "copyOutMs", _lastCopyOutMs.load());

    // Configured delegate + I/O tensor hardware placement (cpu vs delegate).
    obj.setProperty(runtime, "delegate",
                    jsi::String::createFromUtf8(runtime, _fullModel->delegateHardwareName()));
    {
      std::lock_guard<std::mutex> lock(_placementMutex);
      jsi::Array tensorArr(runtime, _placement.size());
      for (size_t i = 0; i < _placement.size(); i++) {
        const auto& t = _placement[i];
        jsi::Object tObj(runtime);
        tObj.setProperty(runtime, "name", jsi::String::createFromUtf8(runtime, t.name));
        tObj.setProperty(runtime, "hardware", jsi::String::createFromUtf8(runtime, t.hardware));
        jsi::Array shapeArr(runtime, t.shape.size());
        for (size_t d = 0; d < t.shape.size(); d++) {
          shapeArr.setValueAtIndex(runtime, d, jsi::Value(static_cast<double>(t.shape[d])));
        }
        tObj.setProperty(runtime, "shape", std::move(shapeArr));
        tensorArr.setValueAtIndex(runtime, i, std::move(tObj));
      }
      obj.setProperty(runtime, "ioTensors", std::move(tensorArr));
    }
    return obj;
  }

  return jsi::HostObject::get(runtime, propNameId);
}

std::vector<jsi::PropNameID> JumpProcessor::getPropertyNames(jsi::Runtime& runtime) {
  std::vector<jsi::PropNameID> result;
  result.push_back(jsi::PropNameID::forAscii(runtime, "pushBackbone"));
  result.push_back(jsi::PropNameID::forAscii(runtime, "setFullModelInterval"));
  result.push_back(jsi::PropNameID::forAscii(runtime, "reset"));
  result.push_back(jsi::PropNameID::forAscii(runtime, "dispose"));
  result.push_back(jsi::PropNameID::forAscii(runtime, "stats"));
  return result;
}

// ---------------------------------------------------------------------------
// Installer
// ---------------------------------------------------------------------------

// Native build marker. Bump this string whenever the native JumpProcessor code
// changes, so JS can verify the DEVICE is actually running the latest native
// binary (dev-pod / NDK caches sometimes link stale object files even after a
// full `expo run`). Read via `global.__jumpProcessorNativeBuild`.
#define JUMP_PROCESSOR_NATIVE_BUILD "2.2.0-feet-rope-channels"

void JumpProcessor::installToRuntime(jsi::Runtime& runtime,
                                     std::shared_ptr<react::CallInvoker> callInvoker) {
  // Expose the native build marker first, so it's available even if anything
  // below were to change.
  runtime.global().setProperty(runtime, "__jumpProcessorNativeBuild",
                               jsi::String::createFromUtf8(runtime, JUMP_PROCESSOR_NATIVE_BUILD));

  auto func = jsi::Function::createFromHostFunction(
      runtime, jsi::PropNameID::forAscii(runtime, "__createJumpProcessor"), 3,
      [callInvoker](jsi::Runtime& rt, const jsi::Value&, const jsi::Value* args,
                    size_t count) -> jsi::Value {
        if (count < 3) {
          throw jsi::JSError(rt, "__createJumpProcessor(fullModel, config, onResult) "
                                 "requires 3 arguments");
        }

        // arg0: the full-model HostObject (a TensorflowPlugin)
        jsi::Object modelObj = args[0].asObject(rt);
        if (!modelObj.isHostObject(rt)) {
          throw jsi::JSError(rt, "__createJumpProcessor: first arg must be a loaded model");
        }
        auto plugin = std::dynamic_pointer_cast<TensorflowPlugin>(modelObj.getHostObject(rt));
        if (plugin == nullptr) {
          throw jsi::JSError(rt, "__createJumpProcessor: first arg is not a TensorflowModel");
        }

        // arg1: config object
        jsi::Object cfgObj = args[1].asObject(rt);
        JumpProcessorConfig config;
        config.bufferSize = readIntProp(rt, cfgObj, "bufferSize");
        config.outputSize = readIntProp(rt, cfgObj, "outputSize");
        config.fullModelInterval = readIntProp(rt, cfgObj, "fullModelInterval");
        config.outputTensorPeriodicity = readIntProp(rt, cfgObj, "outputTensorPeriodicity");
        config.outputTensorPeriod = readIntProp(rt, cfgObj, "outputTensorPeriod");
        // Optional: marks output tensor index (-1 / absent = no marks).
        {
          jsi::Value mv = cfgObj.getProperty(rt, "outputTensorMarks");
          config.outputTensorMarks = mv.isNumber() ? static_cast<int>(mv.asNumber()) : -1;
        }
        // Optional: event-type output tensor index (-1 / absent = no event type).
        {
          jsi::Value ev = cfgObj.getProperty(rt, "outputTensorEventType");
          config.outputTensorEventType = ev.isNumber() ? static_cast<int>(ev.asNumber()) : -1;
        }

        // arg2: onResult callback
        auto onResult = std::make_shared<jsi::Function>(args[2].asObject(rt).asFunction(rt));

        auto processor = std::make_shared<JumpProcessor>(plugin, config, callInvoker, onResult, rt);
        return jsi::Object::createFromHostObject(rt, processor);
      });

  runtime.global().setProperty(runtime, "__createJumpProcessor", func);
}
