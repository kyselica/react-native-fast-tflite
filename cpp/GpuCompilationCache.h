#pragma once

#include <algorithm>
#include <cctype>
#include <exception>
#include <fcntl.h>
#include <filesystem>
#include <string>
#include <sys/file.h>
#include <unistd.h>
#include <vector>

namespace fasttflite {
namespace fs = std::filesystem;

// Serialize disk writers, never camera runtimes. Contention uses uncached GPU
// initialization so a stuck driver cannot block another load's timeout/recovery.
class GpuCacheLease {
public:
  explicit GpuCacheLease(const std::string& directory) {
    if (directory.empty())
      return;
    try {
      auto entry = fs::path(directory);
      auto key = entry.filename().string();
      if (key.size() != 64 || key.find_first_not_of("0123456789abcdef") != std::string::npos)
        return;
      root = entry.parent_path();
      fd = open((root / ".lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
      if (fd < 0 || flock(fd, LOCK_EX | LOCK_NB) != 0)
        return;
      locked = true;
      trim();
      fs::create_directories(entry);
      fs::last_write_time(entry, fs::file_time_type::clock::now());
      path = directory;
    } catch (...) {
      // Cache setup must never prevent an uncached GPU load.
    }
  }
  ~GpuCacheLease() {
    if (locked) {
      try {
        trim();
      } catch (...) {
      }
      flock(fd, LOCK_UN);
    }
    if (fd >= 0)
      close(fd);
  }
  GpuCacheLease(const GpuCacheLease&) = delete;
  GpuCacheLease& operator=(const GpuCacheLease&) = delete;
  const std::string& directory() const {
    return path;
  }
  void invalidate() {
    if (!path.empty()) {
      std::error_code error;
      fs::remove_all(path, error);
    }
  }

private:
  void trim() {
    struct Entry {
      fs::path path;
      fs::file_time_type time;
      uintmax_t bytes;
    };
    std::vector<Entry> entries;
    uintmax_t total = 0;
    for (const auto& entry : fs::directory_iterator(root)) {
      if (!entry.is_directory())
        continue;
      uintmax_t bytes = 0;
      for (const auto& file : fs::recursive_directory_iterator(entry.path())) {
        if (file.is_regular_file())
          bytes += file.file_size();
      }
      entries.push_back({entry.path(), entry.last_write_time(), bytes});
      total += bytes;
    }
    std::sort(entries.begin(), entries.end(),
              [](const Entry& a, const Entry& b) { return a.time < b.time; });
    size_t remaining = entries.size();
    for (const auto& entry : entries) {
      if (remaining <= 4 && total <= 64 * 1024 * 1024)
        break;
      fs::remove_all(entry.path);
      total -= entry.bytes;
      --remaining;
    }
  }
  fs::path root;
  std::string path;
  int fd = -1;
  bool locked = false;
};

inline bool isGpuSerializationFailure(std::string message) {
  std::transform(message.begin(), message.end(), message.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  // Preserve the loader's no-retry policy for allocation errors, even if a
  // driver also mentions serialization in its diagnostic text.
  for (const char* memory : {"memory", "bad_alloc", "allocat"}) {
    if (message.find(memory) != std::string::npos)
      return false;
  }
  // A cold/empty cache is an ordinary miss. In particular, Android's path
  // contains "code_cache" even when the message is only a missing-file notice.
  if (message.find("not found") != std::string::npos ||
      message.find("no serialized data") != std::string::npos)
    return false;
  return message.find("serializ") != std::string::npos;
}

// Factory owns each attempt's resources. A failed attempt is fully destroyed
// before retrying, and no interpreter is ever stored in this disk cache.
template <typename Factory, typename Report>
auto withGpuCache(const std::string& directory, Factory create, Report report) {
  GpuCacheLease cache(directory);
  try {
    return create(cache.directory());
  } catch (const std::exception& error) {
    if (cache.directory().empty() || !isGpuSerializationFailure(error.what()))
      throw;
    cache.invalidate();
    report();
    return create("");
  }
}
} // namespace fasttflite
