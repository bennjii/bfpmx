#define PROFILE 1
#include "profiler/profiler.h"

#include "CPUArithmetic.h"
#include "CPUArithmeticWithoutMarshalling.h"
#include "definition/block_float/block/Block.h"
#include "definition/block_float/block/BlockDims.h"
#include "definition/prelude.h"
#include "helper/test.h"
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <string>

constexpr u32 TestingScalarSize = 4;
using TestingFloat = fp8::E4M3Type;

template <typename Dimensions>
using TestingBlock = Block<TestingScalarSize, Dimensions, TestingFloat,
                           CPUArithmetic, SharedExponentQuantization>;

template <typename Vector, typename ArithmeticImpl>
static void run_arithmetic_test(f64 min, f64 max, size_t iterations) {
  auto _v1 = fill_random_arrays<f64, Vector::NumElems>(min, max);
  auto _v2 = fill_random_arrays<f64, Vector::NumElems>(min, max);
  Vector v1 = Vector::Quantize(_v1);
  Vector v2 = Vector::Quantize(_v2);
  Vector resultTrue, resultNew;
  f64 tolerance = 1.0;
  std::string op = "(op)";

  SECTION("Add") {
    op = "+";
    for (size_t i = 0; i < iterations; i++) {
      {
        profiler::block("naive   add");
        resultTrue = CPUArithmetic<Vector>::Add(v1, v2);
      }
      {
        profiler::block("faster? add");
        resultNew = ArithmeticImpl::Add(v1, v2);
      }
    }
  }

  SECTION("Sub") {
    op = "-";
    for (size_t i = 0; i < iterations; i++) {
      {
        profiler::block("naive   sub");
        resultTrue = CPUArithmetic<Vector>::Sub(v1, v2);
      }
      {
        profiler::block("faster? sub");
        resultNew = ArithmeticImpl::Sub(v1, v2);
      }
    }
  }

  SECTION("Mul") {
    op = "*";
    for (size_t i = 0; i < iterations; i++) {
      {
        profiler::block("naive   mul");
        resultTrue = CPUArithmetic<Vector>::Mul(v1, v2);
      }
      {
        profiler::block("faster? mul");
        resultNew = ArithmeticImpl::Mul(v1, v2);
      }
    }
  }

  SECTION("Div") {
    op = "/";
    _v2 = fill_random_arrays<f64, Vector::NumElems>(std::abs(max / 64),
                                                    std::abs(max));
    v2 = Vector::Quantize(_v2);

    for (size_t i = 0; i < iterations; i++) {
      {
        profiler::block("naive   div");
        resultTrue = CPUArithmetic<Vector>::Div(v1, v2);
      }
      {
        profiler::block("faster? div");
        resultNew = ArithmeticImpl::Div(v1, v2);
      }
    }
  }

  i64 scalar = std::max(resultNew.ScalarBits(), resultTrue.ScalarBits());
  f64 epsilon = std::pow(2, scalar - (i64)Vector::FloatType::SignificandBits());

  REQUIRE(resultNew.Length() == resultTrue.Length());

  for (std::size_t i = 0; i < resultTrue.Length(); i++) {
    bool equal = FuzzyEqual(resultNew.RealizeAtUnsafe(i),
                            resultTrue.RealizeAtUnsafe(i), epsilon * tolerance);

    if (!equal) {
      std::cerr << std::fixed;
      std::cerr << i << ") " << _v1[i] << " " << op << " " << _v2[i] << " => "
                << v1[i] << " " << op << " " << v2[i] << " == " << resultTrue[i]
                << " != " << resultNew[i] << "\n";
    }

    REQUIRE(equal);
  }
}

TEST_CASE("Arithmetic Without Marshalling simulating") {
  profiler::begin();
  using Vector = TestingBlock<BlockDims<10000>>;
  using ar =
      CPUArithmeticWithoutMarshalling<Vector, CPUArithmeticSingularValuesSimulate>;
  run_arithmetic_test<Vector, ar>(-1000.0, 1000.0, 100);
  profiler::end_and_print();
}

TEST_CASE("Arithmetic Without Marshalling floats") {
  profiler::begin();
  using Vector = TestingBlock<BlockDims<10000>>;
  using ar =
      CPUArithmeticWithoutMarshalling<Vector, CPUArithmeticSingularValues>;
  run_arithmetic_test<Vector, ar>(-1000.0, 1000.0, 100);
  profiler::end_and_print();
}
