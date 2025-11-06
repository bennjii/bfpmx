#include "CPUArithmetic.h"
#include "definition/block_float/block/Block.h"
#include "definition/block_float/block/BlockDims.h"
#include "definition/prelude.h"
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iostream>

using TestingDimensions = BlockDims<32>;
using TestingFloat = fp8::E4M3Type;
using TestingBlock = Block<4, TestingDimensions, TestingFloat, CPUArithmetic,
                           SharedExponentQuantization>;

TEST_CASE("GEMV") {

  SECTION("2x3 matrix-vec multiplication") {
    using Matrix2x3 = Block<4, BlockDims<2, 3>, TestingFloat, CPUArithmetic,
                            SharedExponentQuantization>;
    using Vector3 = Block<4, BlockDims<3>, TestingFloat, CPUArithmetic,
                          SharedExponentQuantization>;
    using Vector2 = Block<4, BlockDims<2>, TestingFloat, CPUArithmetic,
                          SharedExponentQuantization>;

    std::array<f64, Matrix2x3::NumElems> matrix_data = {
        2.0, 4.0, 8.0, // Row 0
        1.0, 2.0, 4.0  // Row 1
    };
    std::array<f64, Vector3::NumElems> vector_data = {4.0, 2.0, 1.0};

    Matrix2x3 matrix = Matrix2x3::Quantize(matrix_data);
    Vector3 vector = Vector3::Quantize(vector_data);

    auto result =
        CPUArithmetic<Vector2>::template Gemv<Matrix2x3, Vector3, Vector2>(
            matrix, vector);

    // Expected output: 24 12
    REQUIRE(FuzzyEqual<TestingFloat>(result.RealizeAtUnsafe(0), 24.0)); // row0
    REQUIRE(FuzzyEqual<TestingFloat>(result.RealizeAtUnsafe(1), 12.0)); // row1
  }

  SECTION("3x2 matrix-vec multiplication") {
    using Matrix3x2 = Block<4, BlockDims<3, 2>, TestingFloat, CPUArithmetic,
                            SharedExponentQuantization>;
    using Vector2 = Block<4, BlockDims<2>, TestingFloat, CPUArithmetic,
                          SharedExponentQuantization>;
    using Vector3 = Block<4, BlockDims<3>, TestingFloat, CPUArithmetic,
                          SharedExponentQuantization>;

    std::array<f64, Matrix3x2::NumElems> matrix_data = {4.0, 2.0, 8.0,
                                                        4.0, 2.0, 1.0};
    std::array<f64, Vector2::NumElems> vector_data = {2.0, 4.0};

    Matrix3x2 matrix = Matrix3x2::Quantize(matrix_data);
    Vector2 vector = Vector2::Quantize(vector_data);

    auto result =
        CPUArithmetic<Vector3>::template Gemv<Matrix3x2, Vector2, Vector3>(
            matrix, vector);

    // Expected: 16 38 8
    REQUIRE(FuzzyEqual<TestingFloat>(result.RealizeAtUnsafe(0), 16.0));
    REQUIRE(FuzzyEqual<TestingFloat>(result.RealizeAtUnsafe(1), 32.0));
    REQUIRE(FuzzyEqual<TestingFloat>(result.RealizeAtUnsafe(2), 8.0));
  }
}

TEST_CASE("GEMM") {
  SECTION("3x4 * 4x2 Matrix Multiplication") {
    using Matrix3x4 = Block<4, BlockDims<3, 4>, TestingFloat, CPUArithmetic,
                            SharedExponentQuantization>;
    using Matrix4x2 = Block<4, BlockDims<4, 2>, TestingFloat, CPUArithmetic,
                            SharedExponentQuantization>;
    using Matrix3x2 = Block<4, BlockDims<3, 2>, TestingFloat, CPUArithmetic,
                            SharedExponentQuantization>;

    std::array<f64, 12> a_data = {1.0, 2.0, 4.0, 8.0, 2.0, 4.0,
                                  8.0, 1.0, 4.0, 8.0, 1.0, 2.0};

    std::array<f64, 8> b_data = {2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0};

    Matrix3x4 A = Matrix3x4::Quantize(a_data);
    Matrix4x2 B = Matrix4x2::Quantize(b_data);

    auto result = CPUArithmetic<Matrix3x2>::template Gemm<Matrix3x4, Matrix4x2,
                                                          Matrix3x2>(A, B);

    // Correct result: 20 25 25 20 20 25
    // Actual result: 20 24 24 20 20 24 --[!]--> 25 cannot be represented in
    // fp8::E4M3 so it gets rounded down
    REQUIRE(FuzzyEqual<TestingFloat>(result.RealizeAtUnsafe(0), 20.0));
    REQUIRE(FuzzyEqual<TestingFloat>(result.RealizeAtUnsafe(1), 24.0));
    REQUIRE(FuzzyEqual<TestingFloat>(result.RealizeAtUnsafe(2), 24.0));
    REQUIRE(FuzzyEqual<TestingFloat>(result.RealizeAtUnsafe(3), 20.0));
    REQUIRE(FuzzyEqual<TestingFloat>(result.RealizeAtUnsafe(4), 20.0));
    REQUIRE(FuzzyEqual<TestingFloat>(result.RealizeAtUnsafe(5), 24.0));
  }
}