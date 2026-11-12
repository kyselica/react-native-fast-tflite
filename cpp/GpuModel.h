#pragma once

#include "GpuCompilationCache.h"
#include <cstdarg>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <tflite/delegates/gpu/delegate.h>

// Implemented by the Android adapter. Hashes the exact loaded bytes, not a URL.
std::string fastTfliteGpuCacheDirectory(Buffer buffer, const std::string& configuration);

namespace fasttflite {
struct GpuErrors {
  std::string cacheError;
  bool allocationError = false;
  unsigned int reportedErrors = 0;
  static void record(void* context, const char* format, va_list args) {
    char message[2048];
    vsnprintf(message, sizeof(message), format, args);
    auto* errors = static_cast<GpuErrors*>(context);
    // Classifying delegate failures must not hide the native driver's reason.
    // Bound repeated inference errors while continuing to classify every report.
    if (errors->reportedErrors < 16) {
      ++errors->reportedErrors;
      log("TFLite GPU: %s", message);
    }
    std::string text(message);
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (text.find("memory") != std::string::npos || text.find("allocat") != std::string::npos ||
        text.find("bad_alloc") != std::string::npos)
      errors->allocationError = true;
    if (isGpuSerializationFailure(text))
      errors->cacheError = text;
  }
};

struct GpuResources {
  GpuErrors errors;
  std::string directory;
  std::shared_ptr<TfLiteDelegate> delegate;
};

inline auto loadGpuModel(Buffer buffer, int numThreads, bool fp16,
                         std::shared_ptr<facebook::react::CallInvoker> invoker, bool debug,
                         std::shared_ptr<TensorflowProfilerState> profiler, bool cacheEnabled) {
  // Keep model bytes alive throughout both attempts, including failed creation.
  std::unique_ptr<void, decltype(&free)> bytes(buffer.data, free);
  std::string configuration = std::string("fast-tflite-gpu-v1:litert-1.4.0:") + TfLiteVersion() +
                              ":android-gpu:sustained:min-latency:auto:auto:quant:partitions-1:" +
                              (fp16 ? "fp16:" : "fp32:") + std::to_string(numThreads);
  auto directory =
      cacheEnabled ? fastTfliteGpuCacheDirectory(buffer, configuration) : std::string();
  auto plugin = withGpuCache(
      directory,
      [&](const std::string& cacheDirectory) {
        auto resources = std::make_shared<GpuResources>();
        resources->directory = cacheDirectory;
        auto& errors = resources->errors;
        log("Initializing fresh GPU interpreter (serialization: %s)",
            cacheDirectory.empty() ? "off" : "on");
        auto model = std::unique_ptr<TfLiteModel, decltype(&TfLiteModelDelete)>(
            TfLiteModelCreate(buffer.data, buffer.size), TfLiteModelDelete);
        if (!model)
          throw std::runtime_error("Failed to load TFLite GPU model");
        auto options =
            std::unique_ptr<TfLiteInterpreterOptions, decltype(&TfLiteInterpreterOptionsDelete)>(
                TfLiteInterpreterOptionsCreate(), TfLiteInterpreterOptionsDelete);
        if (!options)
          throw std::runtime_error("Failed to allocate TFLite interpreter options");
        TfLiteInterpreterOptionsSetNumThreads(options.get(), numThreads);
        TfLiteInterpreterOptionsSetErrorReporter(options.get(), GpuErrors::record, &errors);
        auto gpuOptions = TfLiteGpuDelegateOptionsV2Default();
        gpuOptions.is_precision_loss_allowed = fp16 ? 1 : 0;
        gpuOptions.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
        gpuOptions.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
        if (!cacheDirectory.empty()) {
          gpuOptions.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
          gpuOptions.serialization_dir = resources->directory.c_str();
          gpuOptions.model_token = "model";
        }
        auto delegate = std::shared_ptr<TfLiteDelegate>(TfLiteGpuDelegateV2Create(&gpuOptions),
                                                        TfLiteGpuDelegateV2Delete);
        if (!delegate)
          throw std::runtime_error("Failed to create TFLite GPU delegate");
        resources->delegate = delegate;
        TfLiteInterpreterOptionsAddDelegate(options.get(), delegate.get());
        auto interpreter = std::unique_ptr<TfLiteInterpreter, decltype(&TfLiteInterpreterDelete)>(
            TfLiteInterpreterCreate(model.get(), options.get()), TfLiteInterpreterDelete);
        auto failure = [&] {
          if (!errors.allocationError && !errors.cacheError.empty())
            return errors.cacheError;
          return std::string(
              "Failed to allocate memory for input/output tensors in GPU interpreter");
        };
        if (!interpreter) {
          if (!errors.allocationError && errors.cacheError.empty())
            throw std::runtime_error("Failed to create TFLite interpreter");
          throw std::runtime_error(failure());
        }
        auto status = TfLiteInterpreterAllocateTensors(interpreter.get());
        const bool cacheStatus = status == kTfLiteDelegateDataNotFound ||
                                 status == kTfLiteDelegateDataReadError ||
                                 status == kTfLiteDelegateDataWriteError;
        if (!cacheDirectory.empty() && !errors.allocationError &&
            (cacheStatus || !errors.cacheError.empty())) {
          throw std::runtime_error("GPU serialization failed");
        }
        if (status != kTfLiteOk)
          throw std::runtime_error(failure());
        auto result = std::make_shared<TensorflowPlugin>(interpreter.get(), buffer,
                                                         TensorflowPlugin::Delegate::AndroidGPU,
                                                         invoker, debug, profiler, true);
        interpreter.release();
        result->retainGpuResources(resources);
        return result;
      },
      [] { log("GPU serialization unavailable; retrying uncached GPU initialization"); });
  bytes.release();
  return plugin;
}
} // namespace fasttflite
