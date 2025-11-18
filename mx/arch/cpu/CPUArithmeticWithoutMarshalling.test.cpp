#include "CPUArithmetic.h"
#include "CPUArithmeticWithoutMarshalling.h"
#include "definition/block_float/block/Block.h"
#include "definition/block_float/block/BlockDims.h"
#include "definition/prelude.h"
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/interfaces/catch_interfaces_config.hpp>
#include <catch2/internal/catch_context.hpp>
#include <cmath>
#include <iostream>
#include <random>

#define PROFILE 1
#include "../../profiler/profiler.h"

constexpr u32 TestingScalarSize = 4;
using TestingFloat = fp8::E4M3Type;

template <typename Dimensions>
using TestingBlock = Block<TestingScalarSize, Dimensions, TestingFloat,
                           CPUArithmetic, SharedExponentQuantization>;

TEST_CASE("AddFast") {
  SECTION("Vec_10k + Vec_10k") {
    profiler::begin();

    constexpr size_t sz = 10000;
    f64 min = 0.0, max = 1000.0;
    using Vector = TestingBlock<BlockDims<sz>>;

    std::array<f64, Vector::NumElems> _v1;
    std::array<f64, Vector::NumElems> _v2;

    static std::mt19937_64 rng(
        Catch::getCurrentContext().getConfig()->rngSeed());

    std::uniform_real_distribution<f64> dist(-100.0, 100.0);
    for (size_t i = 0; i < sz; i++) {
      _v1[i] = dist(rng);
      _v2[i] = dist(rng);
    }

    Vector v1 = Vector::Quantize(_v1);
    Vector v2 = Vector::Quantize(_v2);

    Vector resultTrue, resultNew;
    {
      profiler::block("naive");
      resultTrue = CPUArithmetic<Vector>::Add(v1, v2);
    }
    {
      profiler::block("no marshal/unmarshal");
      resultNew = CPUArithmeticWithoutMarshalling<Vector>::Add(v1, v2);
    }

    REQUIRE(resultNew.Length() == resultTrue.Length());
    for (std::size_t i = 0; i < resultNew.Length(); i++) {
      if (resultNew.RealizeAtUnsafe(i) != resultTrue.RealizeAtUnsafe(i)) {
        std::cerr << i << " | " << _v1[i] << " + " << _v2[i] << " => "
                  << " | " << v1.RealizeAtUnsafe(i) << " + "
                  << v2.RealizeAtUnsafe(i) << " => "
                  << resultNew.RealizeAtUnsafe(i)
                  << " != " << resultTrue.RealizeAtUnsafe(i);
      }
      REQUIRE(resultNew.RealizeAtUnsafe(i) == resultTrue.RealizeAtUnsafe(i));
    }

    profiler::end_and_print();
  }
}
