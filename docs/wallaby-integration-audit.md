# Wallaby ML integration audit

Audited Wallaby's `t3/reduce-backbone-startup-cost` branch at `319fc4e` and the package source published as `@kyselicaa/react-native-fast-tflite@2.2.2` (`f5aeb34`). The package PR targets `codex/native-model-loader`, which contains that release; the repository's `main` currently predates it.

## Move into the package

- `expo/plugins/withGpuCompilationCache.js` and its `WallabyGpuCache.{h,java}` / `WallabyGpuModel.h` helpers: GPU serialization, content/configuration/runtime invalidation, private cache storage, locking, eviction, cache-only retry, and interpreter/delegate/reporter/model-byte ownership. Implemented directly as `cpp/GpuCompilationCache.h`, `cpp/GpuModel.h` and Android `GpuCompilationCache.java`, with the opt-in `enableGpuCompilationCache` model option.
- `expo/plugins/withJumpProcessorDiagnostics.js` and `JumpProcessorFailureDiagnostics.h`: persistent failure counters and all five native recording sites. The package exposes typed `failureCount` and `lastFailure` in `JumpProcessor.stats`. Counters survive successful inference and `reset()`; reset/generation skips do not count as failures.
- Compiled C++/Java behavior contracts from both plugins. Package tests check native behavior directly rather than installation-time string replacements, source hashes, or dependency version guards.

The extracted functionality depends on native model/processor ownership and backend APIs. Keeping it here avoids patching installed dependency sources and makes its compatibility checks part of the package's own build/release process.

## Keep in Wallaby

- Processor catalogue, Supabase access, model downloads, bundled assets and downloaded-file retention.
- Device-specific delegate/FP16/cadence choices, battery-saver policy and strict benchmark selection.
- Load timeouts and GPU-to-CPU recovery decisions. The package performs only an internal retry for recognized serialization failures.
- Detection attempts, model-load/acquisition/adoption identities, cancellation guards, late-result handling, PostHog events, stats polling and aggregation.
- Camera lifetime, orientation/resize configuration, React providers/hooks and screen navigation.
- Jump/pose/rope analysis and application debug-screen visualizations.
- Whole-app Gradle memory settings and modular-header configuration for other dependencies.

The existing asynchronous native loader is already part of the published package, so no app loader policy needs to move to support this extraction.

## Separate package ownership issue

Wallaby intentionally creates a fresh backbone interpreter per detector mount/rotation because `TensorflowPlugin` output buffers retain JSI objects from their creating worklet runtime. That is ultimately a package ownership concern, but a runtime-aware buffer redesign needs separate native/device validation. Keep the app's fresh-load workaround; compilation caching shares disk artifacts only and does not solve runtime-bound output-buffer reuse.

## After merge and publication

1. Publish a new package version through the maintainer's normal release process. This PR does not change the version or publish anything.
2. Update the app dependency and lockfile to that release and rebuild its native binaries.
3. Remove both app patch plugins, copied native headers/Java source, patch-only tests, plugin registration and postinstall/prebuild patch plumbing. Replace optional runtime probes/casts with the package's typed diagnostics fields where the published binary is guaranteed.
4. Enable `enableGpuCompilationCache` for the intended Android GPU model loads. Preserve runtime selection, fresh per-mount ownership, recovery and cancellation behavior.
5. Keep dependency fingerprint tests, changing them to verify ordinary dependency/native configuration stability rather than asserting the removed patches. Retain app diagnostics/recovery tests and native camera/navigation coverage.
6. Verify cold/warm cache behavior, real inference and output correctness on physical Android OpenCL hardware. Check other GPU backends and iOS for regressions. Contract tests and emulator packaging checks do not establish a speedup.

Wallaby PR [#466](https://github.com/aperturesports/wallaby/pull/466) can then become the dependency/configuration migration. Its existing diagnostics integration should consume the same package release. App changes are deliberately deferred until that release exists.
