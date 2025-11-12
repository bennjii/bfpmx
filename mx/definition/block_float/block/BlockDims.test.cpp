//
// Created by Benjamin White on 10/10/2025.
//

#include "BlockDims.h"

#include <catch2/catch_test_macros.hpp>

#include "definition/prelude.h"

TEST_CASE("BlockDims Construction") { REQUIRE(true); }

TEST_CASE("BlockDims Coordinate Conversions") {
  // 2D Matrix
  using Matrix2x3 = BlockDims<2, 3>;
  const auto matrixTestCases =
      std::to_array<std::pair<std::array<u32, 2>, u32>>({{{0, 0}, 0u},
                                                         {{0, 1}, 1u},
                                                         {{0, 2}, 2u},
                                                         {{1, 0}, 3u},
                                                         {{1, 1}, 4u},
                                                         {{1, 2}, 5u}});

  for (auto [coords, linear] : matrixTestCases) {
    CHECK(Matrix2x3::CoordsToLinear(coords) == linear);
    auto result_coords = Matrix2x3::LinearToCoords(linear);
    CHECK(result_coords[0] == coords[0]);
    CHECK(result_coords[1] == coords[1]);
  }

  // 3D Tensor
  using Tensor2x2x2 = BlockDims<2, 2, 2>;
  const auto tensorTestCases =
      std::to_array<std::pair<std::array<u32, 3>, u32>>({{{0, 0, 0}, 0u},
                                                         {{0, 0, 1}, 1u},
                                                         {{0, 1, 0}, 2u},
                                                         {{0, 1, 1}, 3u},
                                                         {{1, 0, 0}, 4u},
                                                         {{1, 0, 1}, 5u},
                                                         {{1, 1, 0}, 6u},
                                                         {{1, 1, 1}, 7u}});
  for (auto [coords, linear] : tensorTestCases) {
    CHECK(Tensor2x2x2::CoordsToLinear(coords) == linear);
    auto result_coords = Tensor2x2x2::LinearToCoords(linear);
    CHECK(result_coords[0] == coords[0]);
    CHECK(result_coords[1] == coords[1]);
    CHECK(result_coords[2] == coords[2]);
  }
}