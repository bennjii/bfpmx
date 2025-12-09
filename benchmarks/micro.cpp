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

using TestingScalar = u32;
using TestingFloat = fp8::E4M3Type;

template <typename Dimensions>
using TestingBlock = Block<TestingScalar, Dimensions, TestingFloat,
                           CPUArithmetic, SharedExponentQuantization>;

constexpr size_t TestingSize = 10000;
constexpr size_t TestIterations = 100;

using Vector = TestingBlock<BlockDims<TestingSize>>;

constexpr f64 min = 1, max = 100.0;
std::array<f64, Vector::NumElems> v1_;
std::array<f64, Vector::NumElems> v2_;
Vector v1;
Vector v2;

void touch(decltype(v1_) v) {
  f64 sum = 0;
  for (auto &x : v)
    sum += x;
  volatile f64 x = sum;
}

void touch(f64 v) { volatile f64 x = v; }

void touch(decltype(v1) v) {
  f64 sum = 0;
  for (int i = 0; i < v.Length(); i++) {
    sum += v.RealizeAtUnsafe(i);
  }
  volatile f64 x = sum;
}

void TestAll() {
  Vector reference, trial;
  decltype(v1_) real;
  f64 dp;
  {
    profiler::block("normal add");
    for (int j = 0; j < real.size(); j++)
      real[j] = v1_[j] + v2[j];
  }
  touch(real);
  {
    profiler::block("naive add");
    reference = CPUArithmetic<Vector>::Add(v1, v2);
  }
  touch(reference);
  {
    profiler::block("fast-marshal add");
    trial = CPUArithmeticFastMarshalling<Vector>::Add(v1, v2);
  }
  touch(trial);
  {
    profiler::block("normal sub");
    for (int j = 0; j < real.size(); j++)
      real[j] = v1_[j] - v2[j];
  }
  touch(real);
  {
    profiler::block("naive sub");
    reference = CPUArithmetic<Vector>::Sub(v1, v2);
  }
  touch(reference);
  {
    profiler::block("fast-marshal sub");
    trial = CPUArithmeticFastMarshalling<Vector>::Sub(v1, v2);
  }
  touch(trial);
  {
    profiler::block("normal mul");
    for (int j = 0; j < real.size(); j++)
      real[j] = v1_[j] * v2[j];
  }
  touch(real);
  {
    profiler::block("naive mul");
    reference = CPUArithmetic<Vector>::Mul(v1, v2);
  }
  touch(reference);
  {
    profiler::block("fast-marshal mul");
    trial = CPUArithmeticFastMarshalling<Vector>::Mul(v1, v2);
  }
  touch(trial);
  {
    profiler::block("normal div");
    for (int j = 0; j < real.size(); j++)
      real[j] = v1_[j] / v2[j];
  }
  touch(real);
  {
    profiler::block("naive div");
    reference = CPUArithmetic<Vector>::Div(v1, v2);
  }
  touch(reference);
  {
    profiler::block("fast-marshal div");
    trial = CPUArithmeticFastMarshalling<Vector>::Div(v1, v2);
  }
  touch(trial);
  {
    profiler::block("normal DotProduct");
    for (int j = 0; j < real.size(); j++)
      dp += v1_[j] * v2[j];
  }
  touch(dp);
  {
    profiler::block("naive DotProduct");
    dp = CPUArithmetic<Vector>::DotProduct(v1, v2);
  }
  touch(dp);
}

int main(void) {
  profiler::begin();

  auto writer = CsvWriter();
  CsvInfo primitive = PrepareCsvPrimitive("idk", TestingSize, 1);
  CsvInfo block = PrepareCsvBlock<Vector>("idk", TestingSize, 1);

  for (int i = 0; i < TestIterations; i++) {
    v1_ = fill_random_arrays<f64, Vector::NumElems>(min, max);
    v2_ = fill_random_arrays<f64, Vector::NumElems>(min, max);
    v1 = Vector::Quantize(v1_);
    v2 = Vector::Quantize(v2_);

    TestAll();
    writer.next_iteration();
    auto infos = profiler::dump_and_reset();

    for (int j = 0; j < infos.size(); j++) {
      auto const &label = std::string(infos[j].label);
      if (j % 3 == 0) // @hack
        writer.append_csv(primitive, infos[j], 0, 0);
      else
        writer.append_csv(block, infos[j], 0, 0);
    }
  }

  writer.dump("micro.csv");
}
