//
// Created by Benjamin White on 10/10/2025.
//

#include "Block.h"

#include <catch2/catch_test_macros.hpp>

#include "definition/prelude.h"

using TestingDimensions = BlockDims<32>;
using TestingFloat = fp8::E4M3Type;
using TestingBlock = Block<4, TestingDimensions, TestingFloat, CPUArithmetic,
                           SharedExponentQuantization>;

TEST_CASE("Blank Block Construction") {
  const TestingBlock blankBlock;

  // Should all be 0's
  REQUIRE(*blankBlock.RealizeAt(0) == 0.0f);
  REQUIRE(*blankBlock.RealizeAt(31) == 0.0f);

  // Values outside the bounds must be invalid
  REQUIRE(blankBlock.RealizeAt(-1) == std::nullopt);
  REQUIRE(blankBlock.RealizeAt(32) == std::nullopt);
}

TEST_CASE("Block Construction") {
  constexpr std::array<f64, TestingBlock::NumElems> EXAMPLE_ARRAY =
      std::to_array<f64, TestingBlock::NumElems>(
          {1.2f, 3.4f, 5.6f, 2.1f, 1.3f, -6.5f});

  SECTION("construction") {
    const auto QuantizedBlock = TestingBlock::Quantize(EXAMPLE_ARRAY);

    STATIC_REQUIRE(QuantizedBlock.NumDimensions == 1);
    STATIC_REQUIRE(QuantizedBlock.Length() == 32);

    REQUIRE(FuzzyEqual<TestingFloat>(QuantizedBlock.RealizeAtUnsafe(0), 1.2f));
  }
}
