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

void Test(std::string operation, Vector correct, Vector trial) {
  REQUIRE(trial.Length() == correct.Length());

  for (std::size_t i = 0; i < Vector::Length(); i++) {
    bool equal = FuzzyEqual<TestingFloat>(trial[i], correct[i]);

    if (!equal) {
      std::cerr << std::fixed;
      std::cerr << "Assertion failed for element" << i << " of " << Vector::Length() << " elements." << std::endl;
      std::cerr << "Operation: " << operation << std::endl;

      std::cerr << "Raw: "   << v1_[i] << " " << operation << " " << v2_[i] << std::endl;
      std::cerr << "Block: " << v1[i]  << " " << operation << " " << v2[i]  << std::endl;

      std::cerr << "Reference Output: " << correct[i] << std::endl;
      std::cerr << "Actual Output: "    << trial[i]   << std::endl;
    }

    REQUIRE(equal);
  }
}

void TestAll() {
  SECTION("Add") {
    Vector correct, trial;
    {
      profiler::block("naive add");
      correct = CPUArithmetic<Vector>::Add(v1, v2);
    }
    {
      profiler::block("no marshal/unmarshal add");
      trial = CPUArithmeticWithoutMarshalling<Vector>::Add(v1, v2);
    }

    Test("+", correct, trial);
  }

  SECTION("Sub") {
    Vector correct, trial;
    {
      profiler::block("naive sub");
      correct = CPUArithmetic<Vector>::Sub(v1, v2);
    }
    {
      profiler::block("no marshal/unmarshal sub");
      trial = CPUArithmeticWithoutMarshalling<Vector>::Sub(v1, v2);
    }

    Test("-", correct, trial);
  }

  SECTION("Mul") {
    Vector correct, trial;
    {
      profiler::block("naive mul");
      correct = CPUArithmetic<Vector>::Mul(v1, v2);
    }
    {
      profiler::block("no marshal/unmarshal mul");
      trial = CPUArithmeticWithoutMarshalling<Vector>::Mul(v1, v2);
    }

    Test("*", correct, trial);
  }

  SECTION("Div") {
    Vector correct, trial;
    {
      profiler::block("naive div");
      correct = CPUArithmetic<Vector>::Div(v1, v2);
    }
    {
      profiler::block("no marshal/unmarshal div");
      trial = CPUArithmeticWithoutMarshalling<Vector>::Div(v1, v2);
    }

    Test("/", correct, trial);
  }
}

TEST_CASE("Arithmetic Without Marshalling") {
  profiler::begin();

  for (int i = 0; i < TestIterations; i++) {
    TestAll();
  }

  profiler::end_and_print();
}
