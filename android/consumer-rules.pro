# The call-invoker compatibility lookup uses reflection on newer React Native contexts.
-keepclassmembers class com.facebook.react.bridge.ReactContext {
    public *** getJSCallInvokerHolder();
}
-keepclassmembers class * extends com.facebook.react.bridge.ReactContext {
    public *** getJSCallInvokerHolder();
}
