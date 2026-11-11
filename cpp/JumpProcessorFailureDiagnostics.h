#pragma once

#include <atomic>
#include <cstdint>

// lastSkip describes only the most recent run. Keep failures independently so
// a successful run or ring reset between JS snapshots cannot erase an error.
class JumpProcessorFailureDiagnostics {
public:
  void record(int reason) {
    // Ok (0) and reset/generation cancellation (1) are not failures.
    if (reason < 2 || reason > 6)
      return;
    _lastReason.store(reason);
    _count.fetch_add(1);
  }

  uint64_t count() const {
    return _count.load();
  }
  int lastReason() const {
    return _lastReason.load();
  }

private:
  std::atomic<uint64_t> _count{0};
  std::atomic<int> _lastReason{0};
};
