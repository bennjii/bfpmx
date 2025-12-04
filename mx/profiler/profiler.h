/*
 * Created by Gabr1313 on 4/11/2025.
 *
 * Because of the use of __COUNTER__ the profiler works only if used in the same
 compilation unit
 *
 * Usage:
    - `#define PROFILE 1` before including this file.
       Change it to `0` to turn off the profiler.
    - In main function:
          int main(void)
          {
              profiler::begin();
              profiler::end_and_print();
              return 0;
          }
    - To profile something put at the beginning of the block/function this line:
        - profiler::func();
        - profiler::block("block_label");
        - profiler::func_bandwidth(processed_bytes);
        - profiler::block_bandwidth("block_label", processed_bytes)
 */

#ifndef BFPMX_PROFILER_H
#define BFPMX_PROFILER_H

#include "../mx/definition/alias.h"

#include <format>
#include <iostream>

#if defined(_WIN32)

// Windows
#include <intrin.h>
#include <windows.h>

#elif defined(__APPLE__)

// macOS
#include <TargetConditionals.h>

#if TARGET_CPU_X86_64
#include <x86intrin.h>
#elif TARGET_CPU_ARM64
// ARM (Apple Silicon)
#include <arm_neon.h>
#else
#error "Unsupported Apple CPU architecture"
#endif

#include <sys/time.h>

#else

// Linux / other Unix
#include <sys/time.h>

#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#elif defined(__aarch64__) || defined(__arm__)
#include <arm_neon.h>
#else
#warning "Unknown architecture; SIMD unavailable"
#endif

#endif

namespace profiler {
struct ProfilerAnchor {
  u64 elapsed_at_root; // With children
  u64 elapsed_excl;    // Without children
  u64 hit_count;
  u64 byte_count;
  char const *label;
};

namespace detail {
#if _WIN32

static inline u64 get_OS_timer_freq(void) {
  LARGE_INTEGER Freq;
  QueryPerformanceFrequency(&Freq);
  return Freq.QuadPart;
}

static inline u64 read_OS_timer(void) {
  LARGE_INTEGER Value;
  QueryPerformanceCounter(&Value);
  return Value.QuadPart;
}

#else // _WIN32

static inline u64 get_OS_timer_freq(void) { return 1000000; }

static inline u64 read_OS_timer(void) {
  struct timeval Value;
  gettimeofday(&Value, 0);
  return get_OS_timer_freq() * (u64)Value.tv_sec + (u64)Value.tv_usec;
}

#endif // _WIN32

#if defined(__aarch64__)

static inline u64 read_CPU_timer(void) {
  u64 val;
  asm volatile("mrs %0, cntvct_el0" : "=r"(val));
  return val;
}

#elif defined(__x86_64__) || defined(_M_X64)

static inline u64 read_CPU_timer(void) { return __rdtsc(); }

#endif

static inline u64 guess_CPU_freq(u64 milliseconds_to_wait) {
  u64 os_freq = get_OS_timer_freq();
  u64 cpu_start = read_CPU_timer();
  u64 os_start = read_OS_timer();
  u64 os_end = 0;
  u64 os_elapsed = 0;
  u64 os_wait_time = os_freq * milliseconds_to_wait / 1000;
  while (os_elapsed < os_wait_time) {
    os_end = read_OS_timer();
    os_elapsed = os_end - os_start;
  }
  u64 cpu_end = read_CPU_timer();
  u64 cpu_elapsed = cpu_end - cpu_start;
  u64 cpu_freq = 0;
  if (os_elapsed)
    cpu_freq = os_freq * cpu_elapsed / os_elapsed;
  return cpu_freq;
}

#define _TIMINGS_MAX 4096

struct Profiler {
  ProfilerAnchor anchors[_TIMINGS_MAX];
  u32 parent;
};

static Profiler global_profiler; // static -> init to 0
static f64 profiler_cpu_freq;    // static -> init to 0

struct ProfilerBlock;
static inline void
block_end(ProfilerBlock *pb); // forward declaration for the deconstructor
struct ProfilerBlock {
  u64 start;
  u32 anchor_index;
  u32 parent_index;
  u64 elapsed_at_root;

  ~ProfilerBlock() { block_end(this); }
};

static inline void block_end(ProfilerBlock *pb) {
  u64 elapsed = read_CPU_timer() - pb->start;
  ProfilerAnchor *el = &global_profiler.anchors[pb->anchor_index];
  el->elapsed_excl += elapsed;
  global_profiler.anchors[pb->parent_index].elapsed_excl -= elapsed;
  el->elapsed_at_root = pb->elapsed_at_root + elapsed;
  global_profiler.parent = pb->parent_index;
}

static inline f64 to_GbS(u64 elapsed, u64 bytes) {
  f64 seconds = (f64)elapsed / profiler_cpu_freq;
  f64 bytes_per_second = (f64)bytes / seconds;
  f64 megabytes = (f64)bytes / (f64)(1024 * 1024);
  f64 gigabytes_per_second = bytes_per_second / (f64)(1024 * 1024 * 1024);
  return gigabytes_per_second;
}

static inline void do_nothing() {}

#define _concat_inner(a, b) a##b
#define _concat(a, b) _concat_inner(a, b)
#define unique_name(base) _concat(base, __LINE__)
#define block_bandwidth_counter(_block_name, _byte_count, _counter)            \
  detail::ProfilerBlock unique_name(_profiler_) = {};                          \
  static_assert(_counter > 0 && _counter < _TIMINGS_MAX,                       \
                "Too many profiler calls");                                    \
  unique_name(_profiler_).anchor_index = _counter;                             \
  unique_name(_profiler_).parent_index =                                       \
      profiler::detail::global_profiler.parent;                                \
  unique_name(_profiler_).elapsed_at_root =                                    \
      profiler::detail::global_profiler                                        \
          .anchors[unique_name(_profiler_).anchor_index]                       \
          .elapsed_at_root;                                                    \
  profiler::detail::global_profiler.parent =                                   \
      unique_name(_profiler_).anchor_index;                                    \
  profiler::detail::global_profiler                                            \
      .anchors[unique_name(_profiler_).anchor_index]                           \
      .hit_count++;                                                            \
  profiler::detail::global_profiler                                            \
      .anchors[unique_name(_profiler_).anchor_index]                           \
      .label = _block_name;                                                    \
  profiler::detail::global_profiler                                            \
      .anchors[unique_name(_profiler_).anchor_index]                           \
      .byte_count += _byte_count;                                              \
  unique_name(_profiler_).start = profiler::detail::read_CPU_timer()
} // namespace detail

#if PROFILE

static inline void begin(void) {
  detail::profiler_cpu_freq = (f64)detail::guess_CPU_freq(1);
  // here to reset the profiler in case this function is called more than once
  detail::global_profiler = {};
  detail::global_profiler.anchors[0].elapsed_at_root = detail::read_CPU_timer();
}

static inline f64 clocks_to_seconds(u64 clocks) {
  return (f64)clocks / (f64)detail::profiler_cpu_freq;
}

static inline f64 get_elapsed_seconds(const std::string &function_name) {
  for (u32 i = 1; i < _TIMINGS_MAX; i++) {
    if (detail::global_profiler.anchors[i].label == NULL)
      continue;

    if (std::string(detail::global_profiler.anchors[i].label) ==
        function_name) {
      ProfilerAnchor *el = &detail::global_profiler.anchors[i];
      return clocks_to_seconds(el->elapsed_at_root);
    }
  }
  return 0.0;
}

static std::vector<ProfilerAnchor> dump_and_reset(void) {
  detail::global_profiler.anchors[0].elapsed_at_root =
      detail::read_CPU_timer() -
      detail::global_profiler.anchors[0].elapsed_at_root;
  std::vector<ProfilerAnchor> infos;
  for (u32 i = 1;
       // Catch2 or other libraries could use __COUNTER__ too
       i <
       _TIMINGS_MAX /* && detail::global_profiler.anchors[i].label != NULL */;
       i++) {
    // Catch2 or other libraries could use __COUNTER__ too
    if (detail::global_profiler.anchors[i].label == NULL)
      continue;
    ProfilerAnchor *el = &detail::global_profiler.anchors[i];
    infos.push_back(*el);
    *el = {};
  }
  return infos;
}

static inline void end_and_print(void) {
  detail::global_profiler.anchors[0].elapsed_at_root =
      detail::read_CPU_timer() -
      detail::global_profiler.anchors[0].elapsed_at_root;
  f64 total = (f64)detail::global_profiler.anchors[0].elapsed_at_root;
  f64 total_inv = 100.0 / total;
  std::cout << std::format(
                   "[PROFILER] Total time: {:.3f}s (CPU freq guess: {:.2f}Mhz)",
                   total / (f64)detail::profiler_cpu_freq,
                   (f64)detail::profiler_cpu_freq * 1e-6)
            << std::endl;
  for (u32 i = 1;
       // Catch2 or other libraries could use __COUNTER__ too
       i <
       _TIMINGS_MAX /* && detail::global_profiler.anchors[i].label != NULL */;
       i++) {
    // Catch2 or other libraries could use __COUNTER__ too
    if (detail::global_profiler.anchors[i].label == NULL)
      continue;
    ProfilerAnchor *el = &detail::global_profiler.anchors[i];
    std::cout << "[PROFILER]    "
              << std::format("{}[{}] : {} ({:.2f}%, {:.3f}ms", el->label,
                             el->hit_count, el->elapsed_at_root,
                             (f64)el->elapsed_at_root * total_inv,
                             clocks_to_seconds(el->elapsed_at_root) * 1000);
    if (el->elapsed_at_root != el->elapsed_excl)
      std::cout << std::format(", {:.2f}% excl",
                               (f64)el->elapsed_excl * total_inv);
    std::cout << ")";
    if (el->byte_count) {
      f64 megabytes = (f64)el->byte_count / (f64)(1024 * 1024);
      f64 gigabytes_per_second =
          detail::to_GbS(el->elapsed_at_root, el->byte_count);
      std::cout << std::format("  {:.3f}mb at {:.2f}gb/s", megabytes,
                               gigabytes_per_second);
    }
    std::cout << std::endl;
  }
}

#define block_bandwidth(_block_name, _byte_count)                              \
  block_bandwidth_counter(_block_name, _byte_count, __COUNTER__ + 1)
#define func_bandwidth(_byte_count) block_bandwidth(__func__, _byte_count)
#define block(_block_name) block_bandwidth(_block_name, 0)
#define func() func_bandwidth(0)

#else // PROFILE

#ifndef PROFILE
#if _WIN32
#pragma message("including the profiler without previously defining `PROFILE`")
#else
#warning "Including the profiler without previously defining `PROFILE`"
#endif // _WIN32
#endif // ndef PROFILE

#define begin(...) detail::do_nothing()
#define end_and_print(...) detail::do_nothing()
#define block_bandwidth(...) detail::do_nothing()
#define func_bandwidth(...) detail::do_nothing()
#define block(...) detail::do_nothing()
#define func(...) detail::do_nothing()

#endif

} // namespace profiler

#endif // BFPMX_PROFILER_H
