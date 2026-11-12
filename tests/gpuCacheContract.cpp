#include "GpuCompilationCache.h"
#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace facebook::react {
struct CallInvoker {};
} // namespace facebook::react
struct Buffer {
  void* data;
  size_t size;
};
struct TensorflowProfilerState {};
std::vector<std::string> nativeLogs;
void log(const char* format, ...) {
  char message[2048];
  va_list args;
  va_start(args, format);
  vsnprintf(message, sizeof(message), format, args);
  va_end(args);
  nativeLogs.emplace_back(message);
}
using Reporter = void (*)(void*, const char*, va_list);
enum TfLiteStatus {
  kTfLiteOk,
  kTfLiteError,
  kTfLiteDelegateDataNotFound,
  kTfLiteDelegateDataReadError,
  kTfLiteDelegateDataWriteError
};
struct TfLiteModel {};
struct TfLiteDelegate {};
struct TfLiteInterpreterOptions {
  Reporter reporter;
  void* context;
  bool cached = false;
};
struct TfLiteInterpreter {
  TfLiteInterpreterOptions options;
  int id;
};
struct TfLiteGpuDelegateOptionsV2 {
  int is_precision_loss_allowed = 0, inference_preference = 0, inference_priority1 = 0;
  int experimental_flags = 1;
  const char* serialization_dir = nullptr;
  const char* model_token = nullptr;
};
constexpr int TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED = 2;
constexpr int TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY = 1;
constexpr int TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION = 8;
int delegates = 0, interpreters = 0, models = 0, optionsCount = 0, nextId = 0, calls = 0,
    keyCalls = 0;
bool currentCached = false;
std::string mode, cacheDirectory;
std::vector<bool> attempts;
TfLiteModel* TfLiteModelCreate(const void*, size_t) {
  if (mode == "model-creation")
    return nullptr;
  if (mode == "cache" || mode == "gl-fallback" || mode == "cache-then-driver" ||
      mode == "cache-creation") {
    // A failed cached attempt must be completely destroyed before its retry.
    assert(models == 0 && optionsCount == 0 && delegates == 0 && interpreters == 0);
  }
  ++models;
  return new TfLiteModel;
}
void TfLiteModelDelete(TfLiteModel* p) {
  --models;
  delete p;
}
TfLiteInterpreterOptions* TfLiteInterpreterOptionsCreate() {
  if (mode == "options-creation")
    return nullptr;
  ++optionsCount;
  return new TfLiteInterpreterOptions;
}
void TfLiteInterpreterOptionsDelete(TfLiteInterpreterOptions* p) {
  --optionsCount;
  delete p;
}
void TfLiteInterpreterOptionsSetNumThreads(TfLiteInterpreterOptions*, int) {}
void TfLiteInterpreterOptionsSetErrorReporter(TfLiteInterpreterOptions* p, Reporter r, void* c) {
  p->reporter = r;
  p->context = c;
}
void TfLiteInterpreterOptionsAddDelegate(TfLiteInterpreterOptions* p, TfLiteDelegate*) {
  p->cached = currentCached;
}
TfLiteGpuDelegateOptionsV2 TfLiteGpuDelegateOptionsV2Default() {
  return {};
}
TfLiteDelegate* TfLiteGpuDelegateV2Create(const TfLiteGpuDelegateOptionsV2* p) {
  currentCached = p->serialization_dir != nullptr;
  assert(currentCached == bool(p->experimental_flags & 8));
  assert(p->is_precision_loss_allowed == 1);
  if (currentCached)
    assert(std::string(p->model_token) == "model");
  if (mode == "delegate-creation")
    return nullptr;
  ++delegates;
  return new TfLiteDelegate;
}
void TfLiteGpuDelegateV2Delete(TfLiteDelegate* p) {
  if (p == nullptr)
    return;
  --delegates;
  delete p;
}
void report(const TfLiteInterpreterOptions& p, const char* format, ...) {
  va_list args;
  va_start(args, format);
  p.reporter(p.context, format, args);
  va_end(args);
}
TfLiteInterpreter* TfLiteInterpreterCreate(TfLiteModel*, TfLiteInterpreterOptions* p) {
  attempts.push_back(p->cached);
  ++calls;
  if (mode == "creation")
    return nullptr;
  if (mode == "cache-creation" && p->cached) {
    report(*p, "Failed to read serialized data");
    return nullptr;
  }
  ++interpreters;
  return new TfLiteInterpreter{*p, ++nextId};
}
void TfLiteInterpreterDelete(TfLiteInterpreter* p) {
  // Delegates and reporter context must outlive interpreter destruction.
  assert(delegates > 0);
  report(p->options, "interpreter cleanup diagnostic");
  --interpreters;
  delete p;
}
TfLiteStatus TfLiteInterpreterAllocateTensors(TfLiteInterpreter* p) {
  if (p->options.cached &&
      (mode == "cache" || mode == "gl-fallback" || mode == "cache-then-driver")) {
    report(p->options, "Failed to save serialized data");
    return mode == "gl-fallback" ? kTfLiteOk : kTfLiteError;
  }
  if (p->options.cached && mode == "miss") {
    report(p->options, "File /data/user/code_cache/fast-tflite-gpu-v1/model.bin couldn't be opened "
                       "for reading: No such file");
    report(p->options, "No serialized data found: empty.bin");
  }
  if (p->options.cached && mode == "cache-status")
    return kTfLiteDelegateDataWriteError;
  if (mode == "oom-after-cap") {
    for (int i = 0; i < 100; ++i)
      report(p->options, "driver diagnostic GL_INVALID_ENUM");
    report(p->options, "serialization failed: out of memory");
    return kTfLiteError;
  }
  if (mode == "oom")
    report(p->options, "serialization failed: out of memory");
  if (mode == "allocation")
    report(p->options, "tensor allocation failed");
  if (mode == "cache-then-driver" && !p->options.cached)
    return kTfLiteError;
  if (mode == "driver")
    report(p->options, "GPU driver unavailable");
  return mode == "oom" || mode == "allocation" || mode == "driver" ? kTfLiteError : kTfLiteOk;
}
const char* TfLiteVersion() {
  return "2.19.0";
}
struct TensorflowPlugin {
  enum Delegate { AndroidGPU };
  TfLiteInterpreter* interpreter;
  Buffer buffer;
  std::shared_ptr<void> resources;
  TensorflowPlugin(TfLiteInterpreter* p, Buffer b, Delegate,
                   std::shared_ptr<facebook::react::CallInvoker>, bool,
                   std::shared_ptr<TensorflowProfilerState>, bool allocated)
      : interpreter(p), buffer(b) {
    assert(allocated);
  }
  void retainGpuResources(std::shared_ptr<void> r) {
    resources = std::move(r);
  }
  ~TensorflowPlugin() {
    assert(delegates > 0);
    TfLiteInterpreterDelete(interpreter);
    free(buffer.data);
  }
};
#include "GpuModel.h"
std::string fastTfliteGpuCacheDirectory(Buffer, const std::string& config) {
  ++keyCalls;
  assert(config.find("fast-tflite-gpu-v1:litert-1.4.0:2.19.0:android-gpu:") != std::string::npos);
  assert(config.find("fp16:1") != std::string::npos);
  return cacheDirectory;
}
auto load(bool cacheEnabled = true) {
  return fasttflite::loadGpuModel({malloc(128), 128}, 1, true, nullptr, false, nullptr,
                                  cacheEnabled);
}
void noLeaks() {
  assert(interpreters == 0 && delegates == 0 && models == 0 && optionsCount == 0);
}

int main(int argc, char** argv) {
  assert(argc == 2);
  auto root = fasttflite::fs::path(argv[1]);
  fasttflite::fs::create_directories(root);
  cacheDirectory = (root / std::string(64, 'a')).string();
  {
    auto model = load(false);
    assert(keyCalls == 0);
    assert((attempts == std::vector<bool>{false}));
    assert(!fasttflite::fs::exists(cacheDirectory));
  }
  noLeaks();
  attempts.clear();
  {
    auto a = load();
    auto b = load();
    assert(a->interpreter->id != b->interpreter->id);
    assert(a->buffer.data != b->buffer.data);
    assert(a->resources != b->resources);
    assert(delegates == 2 && interpreters == 2);
    // The reporter is still valid after the factory's stack has gone away.
    report(a->interpreter->options, "late inference diagnostic");
  }
  noLeaks();
  // Error classification preserves the native driver message without unbounded
  // per-frame logging. The native reporter remains valid after load returns.
  {
    auto model = load();
    nativeLogs.clear();
    for (int i = 0; i < 100; ++i)
      report(model->interpreter->options, "driver error GL_INVALID_ENUM");
    assert(nativeLogs.size() == 16);
    assert(nativeLogs.front() == "TFLite GPU: driver error GL_INVALID_ENUM");
  }
  noLeaks();
  mode = "miss";
  attempts.clear();
  {
    auto result = load();
    assert((attempts == std::vector<bool>{true}));
  }
  noLeaks();
  for (const auto& cacheFailure : {"cache", "gl-fallback", "cache-status", "cache-creation"}) {
    mode = cacheFailure;
    attempts.clear();
    {
      auto result = load();
      assert((attempts == std::vector<bool>{true, false}));
    }
    noLeaks();
  }
  for (const auto& failure : {"oom", "oom-after-cap", "allocation", "driver", "creation"}) {
    mode = failure;
    attempts.clear();
    try {
      load();
      assert(false);
    } catch (const std::runtime_error&) {
    }
    assert((attempts == std::vector<bool>{true}));
    noLeaks();
  }
  for (const auto& failure : {"model-creation", "options-creation", "delegate-creation"}) {
    mode = failure;
    attempts.clear();
    try {
      load();
      assert(false);
    } catch (const std::runtime_error&) {
    }
    assert(attempts.empty());
    noLeaks();
  }
  mode = "cache-then-driver";
  attempts.clear();
  try {
    load();
    assert(false);
  } catch (const std::runtime_error&) {
  }
  assert((attempts == std::vector<bool>{true, false}));
  noLeaks();
  mode = "";
  attempts.clear();
  {
    fasttflite::GpuCacheLease busy(cacheDirectory);
    auto model = load();
    assert((attempts == std::vector<bool>{false}));
  }
  noLeaks();
  cacheDirectory = (root / "not-a-content-key").string();
  attempts.clear();
  {
    auto model = load();
    assert((attempts == std::vector<bool>{false}));
  }
  noLeaks();
  // LRU and byte bounds are applied after each writer releases the lease.
  for (int i = 0; i < 7; ++i) {
    cacheDirectory = (root / std::string(64, char('0' + i))).string();
    fasttflite::GpuCacheLease cache(cacheDirectory);
    assert(!cache.directory().empty());
  }
  int entries = 0;
  for (auto& entry : fasttflite::fs::directory_iterator(root))
    if (entry.is_directory())
      ++entries;
  assert(entries <= 4);
  {
    fasttflite::GpuCacheLease cache(cacheDirectory);
    FILE* file = fopen((fasttflite::fs::path(cache.directory()) / "large").c_str(), "w");
    assert(file);
    assert(ftruncate(fileno(file), 65 * 1024 * 1024) == 0);
    fclose(file);
  }
  assert(!fasttflite::fs::exists(cacheDirectory));
  assert(!fasttflite::isGpuSerializationFailure("serialization out of memory"));
  assert(!fasttflite::isGpuSerializationFailure("serialization bad_alloc"));
  assert(!fasttflite::isGpuSerializationFailure("cache allocation failed"));
  assert(fasttflite::isGpuSerializationFailure("Failed to SAVE SERIALIZED data"));
}
