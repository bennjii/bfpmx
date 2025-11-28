#include "CPUArithmeticWithoutMarshalling.h"
#include "CPUArithmetic.h"
#include "definition/block_float/block/Block.h"
#include "definition/block_float/block/BlockDims.h"
#include "definition/prelude.h"
#include "helper/test.h"
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <string>

#define PROFILE 1
#include "profiler/profiler.h"

using TestingScalar = u32;
using TestingFloat = fp8::E4M3Type;

template <typename Dimensions>
using TestingBlock = Block<TestingScalar, Dimensions, TestingFloat,
                           CPUArithmetic, SharedExponentQuantization>;

constexpr size_t TestingSize = 10000;
constexpr size_t TestIterations = 100;

using Vector = TestingBlock<BlockDims<TestingSize>>;

constexpr f64 min = -1000.0, max = 1000.0;
auto v1_ = fill_random_arrays<f64, Vector::NumElems>(min, max);
auto v2_ = fill_random_arrays<f64, Vector::NumElems>(min, max);

Vector v1 = Vector::Quantize(v1_);
Vector v2 = Vector::Quantize(v2_);

void Test(std::string operation, Vector reference, Vector trial) {
  REQUIRE(trial.Length() == reference.Length());

  for (std::size_t i = 0; i < Vector::Length(); i++) {
    bool equal = FuzzyEqual<TestingFloat>(trial[i], reference[i]);

    if (!equal) {
      std::cerr << std::fixed;
      std::cerr << "Assertion failed for element" << i << " of " << Vector::Length() << " elements." << std::endl;
      std::cerr << "Operation: " << operation << std::endl;

      std::cerr << "Raw: "   << v1_[i] << " " << operation << " " << v2_[i] << std::endl;
      std::cerr << "Block: " << v1[i]  << " " << operation << " " << v2[i]  << std::endl;

      std::cerr << "Reference Output: " << reference[i] << std::endl;
      std::cerr << "Actual Output: "    << trial[i]   << std::endl;
    }

    REQUIRE(equal);
  }
}

void TestAll() {
  SECTION("Add") {
    Vector reference, trial;
    {
      profiler::block("naive add");
      reference = CPUArithmetic<Vector>::Add(v1, v2);
    }
    {
      profiler::block("no marshal/unmarshal add");
      trial = CPUArithmeticWithoutMarshalling<Vector>::Add(v1, v2);
    }

    Test("+", reference, trial);
  }

  SECTION("Sub") {
    Vector reference, trial;
    {
      profiler::block("naive sub");
      reference = CPUArithmetic<Vector>::Sub(v1, v2);
    }
    {
      profiler::block("no marshal/unmarshal sub");
      trial = CPUArithmeticWithoutMarshalling<Vector>::Sub(v1, v2);
    }

    Test("-", reference, trial);
  }

  SECTION("Mul") {
    Vector reference, trial;
    {
      profiler::block("naive mul");
      reference = CPUArithmetic<Vector>::Mul(v1, v2);
    }
    {
      profiler::block("no marshal/unmarshal mul");
      trial = CPUArithmeticWithoutMarshalling<Vector>::Mul(v1, v2);
    }

    Test("*", reference, trial);
  }

  SECTION("Div") {
    Vector reference, trial;
    {
      profiler::block("naive div");
      reference = CPUArithmetic<Vector>::Div(v1, v2);
    }
    {
      profiler::block("no marshal/unmarshal div");
      trial = CPUArithmeticWithoutMarshalling<Vector>::Div(v1, v2);
    }

    Test("/", reference, trial);
  }
}

TEST_CASE("Arithmetic Without Marshalling") {
  profiler::begin();

  for (int i = 0; i < TestIterations; i++) {
    TestAll();
  }

  profiler::end_and_print();
}
