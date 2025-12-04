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

static CsvInfo CsvInfo_f32(std::string stress_function, u64 input_size,
                           u64 steps) {
  CsvInfo ret = {};
  ret.format = "f32";
  ret.block_size = 0;
  ret.policy = "";
  ret.stress_function = stress_function;
  ret.input_size = input_size;
  ret.steps = steps;
  return ret;
}

static CsvInfo CsvInfo_mx(std::string stress_function, u64 input_size,
                          u64 steps, std::string format, u64 block_size,
                          std::string policy) {
  CsvInfo ret = {};
  ret.format = format;
  ret.block_size = block_size;
  ret.policy = policy;
  ret.stress_function = stress_function;
  ret.input_size = input_size;
  ret.steps = steps;
  return ret;
}

// template <T>
static CsvInfo CsvInfo_mx(std::string stress_function, u64 input_size,
                          u64 steps) {
  // TODO: read values from T = Block<....>
  //       sincerily I need help
  return CsvInfo_mx(stress_function, input_size, steps, "" /*TODO*/, 0 /*TODO*/,
                    "" /*TODO*/);
}

class CsvWriter {
public:
  CsvWriter() {
    csv = "format, block_size, policy, stress_function, input_size, "
          "steps, iteration_id, label, clocks_inclusive, "
          "clocks_exclusive, hit_count, error\n";
    iteration_id = -1;
  }

  u64 next_iteration() { return ++iteration_id; }

  void append_csv(CsvInfo const &basic_info,
                  profiler::ProfilerAnchor const &profiler_info,
                  f64 error /* TODO @benji */) {
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
    csv += std::to_string(iteration_id);
    csv += ", ";
    csv += std::string(profiler_info.label);
    csv += ", ";
    csv += std::to_string(profiler_info.elapsed_at_root);
    csv += ", ";
    csv += std::to_string(profiler_info.elapsed_excl);
    csv += ", ";
    csv += std::to_string(profiler_info.hit_count);
    csv += ", ";
    csv += std::to_string(error);
    csv += "\n";
  }

  void dump(std::ostream &out) const { out << csv; }

  void dump(std::string file_name) const {
    std::ofstream file("benchmarks/csv/jacobi-2d.csv");
    dump(file);
  }

private:
  i64 iteration_id;
  std::string csv;
};

#endif // BFPMX_CSV_INFO_H
