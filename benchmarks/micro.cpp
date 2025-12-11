#include <cstddef>
#define PROFILE 1
#include "profiler/csv_info.h"
#include "profiler/profiler.h"

#include "arch/cpu/CPUArithmetic.h"
#include "arch/cpu/CPUArithmeticFastMarshalling.h"
#include "definition/block_float/block/Block.h"
#include "definition/block_float/block/BlockDims.h"
#include "definition/prelude.h"
#include "helper/test.h"
#include <array>
#include <string>

using TestingFloat = fp8::E4M3Type;
template <typename Dimensions>
using TestingBlock = Block<u32, Dimensions, TestingFloat, CPUArithmetic,
                           SharedExponentQuantization>;

template <size_t TestingSize> void TestAll(f64 min, f64 max) {
  using Vector = TestingBlock<BlockDims<TestingSize>>;

  std::array<f64, Vector::NumElems> v1_ =
      fill_random_arrays<f64, Vector::NumElems>(min, max);
  std::array<f64, Vector::NumElems> v2_ =
      fill_random_arrays<f64, Vector::NumElems>(min, max);
  Vector v1 = Vector::Quantize(v1_);
  Vector v2 = Vector::Quantize(v2_);

  auto touch_array = [&](std::array<f64, Vector::NumElems> const &v) {
    f64 sum = 0;
    for (auto &x : v)
      sum += x;
    volatile f64 x = sum;
  };

  auto touch_f64 = [&](f64 v) { volatile f64 x = v; };

  auto touch_mx = [&](Vector const &v) {
    f64 sum = 0;
    for (int i = 0; i < v.Length(); i++) {
      sum += v.RealizeAtUnsafe(i);
    }
    volatile f64 x = sum;
  };

  Vector reference, trial;
  std::array<f64, Vector::NumElems> real;
  f64 dp;
  {
    profiler::block("normal add");
    for (int j = 0; j < real.size(); j++)
      real[j] = v1_[j] + v2[j];
  }
  touch_array(real);
  {
    profiler::block("naive add");
    reference = CPUArithmetic<Vector>::Add(v1, v2);
  }
  touch_mx(reference);
  {
    profiler::block("fast-marshal add");
    trial = CPUArithmeticFastMarshalling<Vector>::Add(v1, v2);
  }
  touch_mx(trial);
  {
    profiler::block("normal sub");
    for (int j = 0; j < real.size(); j++)
      real[j] = v1_[j] - v2[j];
  }
  touch_array(real);
  {
    profiler::block("naive sub");
    reference = CPUArithmetic<Vector>::Sub(v1, v2);
  }
  touch_mx(reference);
  {
    profiler::block("fast-marshal sub");
    trial = CPUArithmeticFastMarshalling<Vector>::Sub(v1, v2);
  }
  touch_mx(trial);
  {
    profiler::block("normal mul");
    for (int j = 0; j < real.size(); j++)
      real[j] = v1_[j] * v2[j];
  }
  touch_array(real);
  {
    profiler::block("naive mul");
    reference = CPUArithmetic<Vector>::Mul(v1, v2);
  }
  touch_mx(reference);
  {
    profiler::block("fast-marshal mul");
    trial = CPUArithmeticFastMarshalling<Vector>::Mul(v1, v2);
  }
  touch_mx(trial);
  {
    profiler::block("normal div");
    for (int j = 0; j < real.size(); j++)
      real[j] = v1_[j] / v2[j];
  }
  touch_array(real);
  {
    profiler::block("naive div");
    reference = CPUArithmetic<Vector>::Div(v1, v2);
  }
  touch_mx(reference);
  {
    profiler::block("fast-marshal div");
    trial = CPUArithmeticFastMarshalling<Vector>::Div(v1, v2);
  }
  touch_mx(trial);
  {
    profiler::block("normal DotProduct");
    for (int j = 0; j < real.size(); j++)
      dp += v1_[j] * v2[j];
  }
  touch_f64(dp);
  {
    profiler::block("naive DotProduct");
    dp = CPUArithmetic<Vector>::DotProduct(v1, v2);
  }
  touch_f64(dp);
}

template <size_t... Sizes> struct SizeList {};

template <size_t TestingSize>
void RunOneSizeTest(CsvWriter &writer, size_t test_iterations, f64 min,
                    f64 max) {
  for (int i = 0; i < test_iterations; i++) {
    CsvInfo primitive = PrepareCsvPrimitive("---", TestingSize, 1);
    CsvInfo block = PrepareCsvBlock<TestingBlock<BlockDims<TestingSize>>>(
        "---", TestingSize, 1);
    TestAll<TestingSize>(min, max);
    writer.next_iteration();
    auto infos = profiler::dump_and_reset();
    for (int j = 0; j < infos.size(); j++) {
      auto const &label = std::string(infos[j].label);
      if (j % 3 == 0) // @hack
        writer.append_csv(primitive, infos[j], -1, -1);
      else
        writer.append_csv(block, infos[j], -1, -1);
    }
  }
}

template <size_t... Sizes>
void RunAllTests(SizeList<Sizes...>, size_t test_iterations, f64 min, f64 max) {
  auto writer = CsvWriter();
  (RunOneSizeTest<Sizes>(writer, test_iterations, min, max), ...);
  writer.dump("micro.csv");
}

int main(void) {
  profiler::begin();
  size_t test_iterations = 100;
  f64 min = 1, max = 100;
  using list = SizeList<10, 100, 1000, 10000>;
  RunAllTests(list{}, test_iterations, min, max);
}
