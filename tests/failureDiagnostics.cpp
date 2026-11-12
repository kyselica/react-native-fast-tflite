#include "JumpProcessorFailureDiagnostics.h"
#include <cassert>
#include <initializer_list>
#include <thread>

int main() {
  JumpProcessorFailureDiagnostics diagnostics;
  assert(diagnostics.count() == 0);
  assert(diagnostics.lastReason() == 0);

  // Polling after an intervening success must still expose the earlier failure.
  std::atomic<int> lastSkip{0};
  std::thread worker([&] {
    lastSkip.store(4);
    diagnostics.record(4);
    lastSkip.store(0);
  });
  worker.join();
  assert(lastSkip.load() == 0);
  assert(diagnostics.count() == 1);
  assert(diagnostics.lastReason() == 4);

  for (int reason : {-1, 0, 1, 7}) {
    diagnostics.record(reason);
  }
  assert(diagnostics.count() == 1);
  assert(diagnostics.lastReason() == 4);

  for (int reason = 2; reason <= 6; ++reason) {
    diagnostics.record(reason);
    assert(diagnostics.count() == static_cast<uint64_t>(reason));
    assert(diagnostics.lastReason() == reason);
  }

  std::thread invoke([&] {
    for (int i = 0; i < 1000; ++i)
      diagnostics.record(4);
  });
  std::thread delivery([&] {
    for (int i = 0; i < 1000; ++i)
      diagnostics.record(6);
  });
  invoke.join();
  delivery.join();
  assert(diagnostics.count() == 2006);
  assert(diagnostics.lastReason() == 4 || diagnostics.lastReason() == 6);
}
