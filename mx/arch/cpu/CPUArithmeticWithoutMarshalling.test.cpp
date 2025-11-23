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
    constexpr size_t sz = 10000;
    constexpr size_t iterations = 100;
    f64 min = -1000.0, max = 1000.0;

    using Vector = TestingBlock<BlockDims<sz>>;
    auto _v1 = fill_random_arrays<f64, Vector::NumElems>(min, max);
    auto _v2 = fill_random_arrays<f64, Vector::NumElems>(min, max);

    // TODO: wrong test in CPUArithmetic because the 
    //       block Bias is different => different precision
    //       we have to fix FuzzyEqual to consider the exponent of the biggest compared number...
    // _v1[0] = 14.3838;
    // _v2[0] = 1.25202;
    // _v1[1] = 0.0;
    // _v2[2] = 1000.0;

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
                  << v1[i].value() << " + " << v2[i].value()
                  << " == " << resultTrue[i].value()
                  << " != " << resultNew[i].value() << "\n";
      }
      REQUIRE(resultNew.RealizeAtUnsafe(i) == resultTrue.RealizeAtUnsafe(i));
    }
  }

  SECTION("Sub") {
    constexpr size_t sz = 10000;
    constexpr size_t iterations = 100;
    f64 min = -1000.0, max = 1000.0;

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
                  << v1[i].value() << " + " << v2[i].value()
                  << " == " << resultTrue[i].value()
                  << " != " << resultNew[i].value() << "\n";
      }
      REQUIRE(resultNew.RealizeAtUnsafe(i) == resultTrue.RealizeAtUnsafe(i));
    }
  }
}
