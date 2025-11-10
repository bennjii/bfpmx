//
// Created by Benjamin White on 10/10/2025.
//

#include <catch2/catch_test_macros.hpp>

#include "definition/prelude.h"

using TestingDimensions = BlockDims<32>;
using TestingFloat = fp8::E4M3Type;

template <template <std::size_t, BlockDimsType,
                    IFloatRepr, template<typename> typename ArithmeticPolicy_> typename QuantizationPolicy>
using TestingBlock = Block<4, TestingDimensions, TestingFloat, CPUArithmetic,
                           QuantizationPolicy>;

std::array<f64, TestingDimensions::TotalSize()> EXAMPLE_ARRAY =
    std::to_array<f64, TestingDimensions::TotalSize()>(
        {1.2f, 3.4f, 5.6f, 2.1f, 1.3f, -6.5f});

TEST_CASE("Maximum Fractional Quantization") {
  using MaximumFractionalBlock = TestingBlock<MaximumFractionalQuantization>;
  MaximumFractionalBlock block =
      MaximumFractionalBlock::Quantize(EXAMPLE_ARRAY);

  // Should match all from original array, within some given bound
  for (u32 i = 0; i < MaximumFractionalBlock::NumElems; i++) {
    REQUIRE(FuzzyEqual<TestingFloat>(block[i].value(), EXAMPLE_ARRAY[i]));
  }
}

TEST_CASE("Shared Exponent Quantization") {
  using SharedExponentBlock = TestingBlock<SharedExponentQuantization>;
  SharedExponentBlock block = SharedExponentBlock::Quantize(EXAMPLE_ARRAY);

  // Should match all from original array, within some given bound
  for (u32 i = 0; i < SharedExponentBlock::NumElems; i++) {
    REQUIRE(FuzzyEqual<TestingFloat>(block[i].value(), EXAMPLE_ARRAY[i]));
  }
}