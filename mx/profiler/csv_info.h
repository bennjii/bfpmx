/*
 * Created by Gabr1313 on 3/12/2025.
 */

#ifndef BFPMX_CSV_INFO_H
#define BFPMX_CSV_INFO_H

#include "prelude.h"
#include "profiler/profiler.h"
#include <fstream>
#include <string>

struct CsvInfo {
  std::string format;
  u64 block_size;
  std::string policy;
  std::string stress_function;
  u64 input_size;
  u64 steps;
};

static CsvInfo PrepareCsvPrimitive(std::string const &stress_function,
                                   const u64 input_size, const u64 steps) {
  CsvInfo ret = {};

  ret.format = "primitive";
  ret.block_size = 0;
  ret.policy = "";
  ret.stress_function = stress_function;
  ret.input_size = input_size;
  ret.steps = steps;

  return ret;
}

template <typename Block>
static CsvInfo PrepareCsvBlock(std::string const &stress_function,
                               const u64 input_size, const u64 steps) {
  CsvInfo ret = {};
  ret.format = Block::FloatType::Nomenclature();
  ret.block_size = Block::Length();
  ret.policy = Block::QuantizationPolicyType::Identity();
  ret.stress_function = stress_function;
  ret.input_size = input_size;
  ret.steps = steps;
  return ret;
}

class CsvWriter {
public:
  CsvWriter() {
    csv = "format, block_size, policy, stress_function, input_size, "
          "steps, iteration_id, label, clocks_inclusive, "
          "clocks_exclusive, runtime (ms), hit_count, error (%), error (+/-)\n";
    iteration_id = -1;
  }

  u64 next_iteration() { return ++iteration_id; }

  void write_err_only(CsvInfo const &basic_info, std::string const &label,
                      const f64 iteration, const f64 error_percent,
                      const f64 error_abs) {
    write_line(basic_info, label, -1, -1, -1, iteration, -1, error_percent,
               error_abs);
  }

  void write_line(CsvInfo const &basic_info, std::string const &label,
                  const i64 elapsed, const i64 elapsed_exclusive,
                  const i64 hit_count, const i64 iteration,
                  const f64 runtime_ms, const f64 error_percent,
                  const f64 error_abs) {
    csv += basic_info.format;
    csv += ", ";
    csv += std::to_string(basic_info.block_size);
    csv += ", ";
    csv += basic_info.policy;
    csv += ", ";
    csv += basic_info.stress_function;
    csv += ", ";
    csv += std::to_string(basic_info.input_size);
    csv += ", ";
    csv += std::to_string(basic_info.steps);
    csv += ", ";
    csv += std::to_string(iteration);
    csv += ", ";
    csv += std::string(label);
    csv += ", ";
    csv += std::to_string(elapsed);
    csv += ", ";
    csv += std::to_string(elapsed_exclusive);
    csv += ", ";
    csv += std::to_string(runtime_ms);
    csv += ", ";
    csv += std::to_string(hit_count);
    csv += ", ";
    csv += std::to_string(error_percent);
    csv += ", ";
    csv += std::to_string(error_abs);
    csv += "\n";
  }

  void append_csv(CsvInfo const &basic_info,
                  profiler::ProfilerAnchor const &profiler_info,
                  const f64 error_percent, const f64 error_abs) {
    const auto runtime_ms =
        profiler::clocks_to_seconds(profiler_info.elapsed_at_root) * 1000;

    write_line(basic_info, profiler_info.label, profiler_info.elapsed_at_root,
               profiler_info.elapsed_excl, profiler_info.hit_count,
               iteration_id, runtime_ms, error_percent, error_abs);
  }

  void dump(std::ostream &out) const { out << csv; }

  void dump(std::string const &file_name) const {
    std::ofstream file(file_name);
    dump(file);
  }

private:
  i64 iteration_id;
  std::string csv;
};

#endif // BFPMX_CSV_INFO_H
