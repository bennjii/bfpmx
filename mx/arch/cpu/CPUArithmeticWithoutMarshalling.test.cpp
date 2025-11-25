#include "CPUArithmeticWithoutMarshalling.h"
#include "CPUArithmetic.h"
#include "definition/block_float/block/Block.h"
#include "definition/block_float/block/BlockDims.h"
#include "definition/prelude.h"
#include "helper/test.h"
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <iostream>

#define PROFILE 1
#include "profiler/profiler.h"

constexpr u32 TestingScalarSize = 4;
using TestingFloat = fp8::E4M3Type;

template <typename Dimensions>
using TestingBlock = Block<TestingScalarSize, Dimensions, TestingFloat,
                           CPUArithmetic, SharedExponentQuantization>;

TEST_CASE("Arithmetic Without Marshalling") {
  SECTION("Add") {
    constexpr size_t sz = 1000;
    constexpr size_t iterations = 1000;
    f64 min = 0.0, max = 1000.0;

    using Vector = TestingBlock<BlockDims<sz>>;
    auto _v1 = fill_random_arrays<f64, Vector::NumElems>(min, max);
    auto _v2 = fill_random_arrays<f64, Vector::NumElems>(min, max);

    Vector v1 = Vector::Quantize(_v1);
    Vector v2 = Vector::Quantize(_v2);

    Vector resultTrue, resultNew;
    profiler::begin();
    for (size_t i = 0; i < iterations; i++) {
      {
        profiler::block("naive");
        resultTrue = CPUArithmetic<Vector>::Add(v1, v2);
      }
      {
        profiler::block("no marshal/unmarshal");
        resultNew = CPUArithmeticWithoutMarshalling<Vector>::Add(v1, v2);
      }
    }
    profiler::end_and_print();

    REQUIRE(resultNew.Length() == resultTrue.Length());
    for (std::size_t i = 0; i < resultTrue.Length(); i++) {
      if (resultNew.RealizeAtUnsafe(i) != resultTrue.RealizeAtUnsafe(i)) {
        std::cerr << i << " | " << _v1[i] << " + " << _v2[i] << " => " << " | "
                  << v1[i] << " + " << v2[i] << " ==" << resultTrue[i]
                  << " != " << resultNew[i];
      }
      REQUIRE(resultNew.RealizeAtUnsafe(i) == resultTrue.RealizeAtUnsafe(i));
    }
  }
}
