#ifdef RCT_NEW_ARCH_ENABLED

#import "RNTfliteSpec.h"
#import <ReactCommon/RCTTurboModuleWithJSIBindings.h>
@interface Tflite : NSObject <NativeRNTfliteSpec, RCTTurboModuleWithJSIBindings>
@end

#else

#import <React/RCTBridgeModule.h>
@interface Tflite : NSObject <RCTBridgeModule>
@end

#endif
