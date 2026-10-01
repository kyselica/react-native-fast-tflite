#import "Tflite.h"
#import "../cpp/JumpProcessor.h"
#import "../cpp/TensorflowPlugin.h"
#import <React-callinvoker/ReactCommon/CallInvoker.h>
#ifndef RCT_NEW_ARCH_ENABLED
#import <React/RCTBridge+Private.h>
#endif
#import <jsi/jsi.h>
#import <string>

#ifndef RCT_NEW_ARCH_ENABLED
@interface RCTBridge (RCTTurboModule)
- (std::shared_ptr<facebook::react::CallInvoker>)jsCallInvoker;
@end
#endif

using namespace facebook;

@implementation Tflite {
  BOOL _bindingsInstalled;
}
RCT_EXPORT_MODULE(Tflite)

- (BOOL)installBindingsWithRuntime:(jsi::Runtime &)runtime
                      callInvoker:(const std::shared_ptr<react::CallInvoker> &)callInvoker {
  auto fetchByteDataFromUrl = [](std::string url) {
    NSString* string = [NSString stringWithUTF8String:url.c_str()];
    NSLog(@"Fetching %@...", string);
    NSURL* nsURL = [NSURL URLWithString:string];
    NSData* contents = [NSData dataWithContentsOfURL:nsURL];

    void* data = malloc(contents.length * sizeof(uint8_t));
    memcpy(data, contents.bytes, contents.length);
    return Buffer{.data = data, .size = contents.length};
  };

  try {
    TensorflowPlugin::installToRuntime(runtime, callInvoker, fetchByteDataFromUrl);
    JumpProcessor::installToRuntime(runtime, callInvoker);
  } catch (std::exception& exc) {
    NSLog(@"Failed to install TensorFlow Lite plugin to Runtime! %s", exc.what());
    return NO;
  }

  return YES;
}

- (NSNumber *)install {
#ifdef RCT_NEW_ARCH_ENABLED
  // React Native installs bindings when it creates this TurboModule, before JS calls install().
  return @(_bindingsInstalled);
#else
  RCTBridge *bridge = [RCTBridge currentBridge];
  RCTCxxBridge *cxxBridge = (RCTCxxBridge *)bridge;
  if (!cxxBridge.runtime) {
    return @(NO);
  }
  jsi::Runtime &runtime = *(jsi::Runtime *)cxxBridge.runtime;
  return @([self installBindingsWithRuntime:runtime callInvoker:[bridge jsCallInvoker]]);
#endif
}

#ifdef RCT_NEW_ARCH_ENABLED
- (void)installJSIBindingsWithRuntime:(jsi::Runtime &)runtime
                        callInvoker:(const std::shared_ptr<react::CallInvoker> &)callInvoker {
  _bindingsInstalled = [self installBindingsWithRuntime:runtime callInvoker:callInvoker];
}

- (std::shared_ptr<facebook::react::TurboModule>)getTurboModule:
    (const facebook::react::ObjCTurboModule::InitParams&)params {
  return std::make_shared<facebook::react::NativeRNTfliteSpecJSI>(params);
}
#endif

@end
