//
//  DelegateWrapper.mm
//  react-native-fast-tflite
//
//  Objective-C++ implementation of delegate wrappers.
//  This file must be compiled as Objective-C++ (.mm) because the delegate
//  framework headers may import Objective-C frameworks like Metal or CoreML.
//

#import <TensorFlowLiteC/TensorFlowLiteC.h>

#if FAST_TFLITE_ENABLE_METAL
#import <TensorFlowLiteCMetal/TensorFlowLiteCMetal.h>
#endif

#if FAST_TFLITE_ENABLE_CORE_ML
#import <TensorFlowLiteCCoreML/TensorFlowLiteCCoreML.h>
#endif

extern "C" {

// MARK: - Metal Delegate

bool TFLIsMetalDelegateAvailable(void) {
#if FAST_TFLITE_ENABLE_METAL
  return true;
#else
  return false;
#endif
}

TfLiteDelegate* TFLCreateMetalDelegate(bool allowPrecisionLoss) {
#if FAST_TFLITE_ENABLE_METAL
  // FP32 by default (FP16 gave negligible speedup on Apple GPUs). The benchmark
  // dev screen can opt into FP16 (allow_precision_loss) to A/B it.
  TFLGpuDelegateOptions options = TFLGpuDelegateOptionsDefault();
  options.allow_precision_loss = allowPrecisionLoss;
  return TFLGpuDelegateCreate(&options);
#else
  (void)allowPrecisionLoss;
  return nullptr;
#endif
}

void TFLDeleteMetalDelegate(TfLiteDelegate* delegate) {
#if FAST_TFLITE_ENABLE_METAL
  if (delegate != nullptr) {
    TFLGpuDelegateDelete(delegate);
  }
#endif
}

// MARK: - CoreML Delegate

bool TFLIsCoreMLDelegateAvailable(void) {
#if FAST_TFLITE_ENABLE_CORE_ML
  return true;
#else
  return false;
#endif
}

TfLiteDelegate* TFLCreateCoreMLDelegate(void) {
#if FAST_TFLITE_ENABLE_CORE_ML
  TfLiteCoreMlDelegateOptions options = {.enabled_devices = TfLiteCoreMlDelegateAllDevices,
                                         .coreml_version = 0,
                                         .max_delegated_partitions = 0,
                                         .min_nodes_per_partition = 2};
  return TfLiteCoreMlDelegateCreate(&options);
#else
  return nullptr;
#endif
}

void TFLDeleteCoreMLDelegate(TfLiteDelegate* delegate) {
#if FAST_TFLITE_ENABLE_CORE_ML
  if (delegate != nullptr) {
    TfLiteCoreMlDelegateDelete(delegate);
  }
#endif
}

} // extern "C"
