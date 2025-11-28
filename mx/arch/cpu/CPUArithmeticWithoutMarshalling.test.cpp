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

TEST_CASE("Arithmetic Without Marshalling") {
  constexpr size_t sz = 10000;
  constexpr size_t iterations = 100;
  f64 min = -1000.0, max = 1000.0;

  using Vector = TestingBlock<BlockDims<sz>>;
  auto _v1 = fill_random_arrays<f64, Vector::NumElems>(min, max);
  auto _v2 = fill_random_arrays<f64, Vector::NumElems>(min, max);
  Vector v1 = Vector::Quantize(_v1);
  Vector v2 = Vector::Quantize(_v2);
  Vector resultTrue, resultNew;
  std::string op = "(op)";
  f64 tollerance = 4;

  profiler::begin();
  SECTION("Add") {
    op = "+";
    tollerance = 4;
    for (size_t i = 0; i < iterations; i++) {
      {
        profiler::block("naive add");
        resultTrue = CPUArithmetic<Vector>::Add(v1, v2);
      }
      {
        profiler::block("no marshal/unmarshal add");
        resultNew = CPUArithmeticWithoutMarshalling<Vector>::Add(v1, v2);
      }
    }
  }

  SECTION("Sub") {
    op = "-";
    tollerance = 4;
    for (size_t i = 0; i < iterations; i++) {
      {
        profiler::block("naive sub");
        resultTrue = CPUArithmetic<Vector>::Sub(v1, v2);
      }
      {
        profiler::block("no marshal/unmarshal sub");
        resultNew = CPUArithmeticWithoutMarshalling<Vector>::Sub(v1, v2);
      }
    }
  }

  SECTION("Mul") {
    op = "*";
    tollerance = 4;
    for (size_t i = 0; i < iterations; i++) {
      {
        profiler::block("naive mul");
        resultTrue = CPUArithmetic<Vector>::Mul(v1, v2);
      }
      {
        profiler::block("no marshal/unmarshal mul");
        resultNew = CPUArithmeticWithoutMarshalling<Vector>::Mul(v1, v2);
      }
    }
  }

  SECTION("Div") {
    op = "/";
    tollerance = 2;
    _v2 = fill_random_arrays<f64, Vector::NumElems>(std::abs(max / 64),
                                                    std::abs(max));
    v2 = Vector::Quantize(_v2);
    for (size_t i = 0; i < iterations; i++) {
      {
        profiler::block("naive div");
        resultTrue = CPUArithmetic<Vector>::Div(v1, v2);
      }
      {
        profiler::block("no marshal/unmarshal div");
        resultNew = CPUArithmeticWithoutMarshalling<Vector>::Div(v1, v2);
      }
    }
  }

  profiler::end_and_print();

  REQUIRE(resultNew.Length() == resultTrue.Length());
  for (std::size_t i = 0; i < resultTrue.Length(); i++) {
    bool equal =
        FuzzyEqual<TestingFloat>(resultNew.RealizeAtUnsafe(i),
                                 resultTrue.RealizeAtUnsafe(i), tollerance);
    if (!equal) {
      std::cerr << std::fixed;
      std::cerr << i << ") " << _v1[i] << " " << op << " " << _v2[i] << " => "
                << v1[i] << " " << op << " " << v2[i] << " == " << resultTrue[i]
                << " != " << resultNew[i] << "\n";
    }
    REQUIRE(equal);
  }
}
