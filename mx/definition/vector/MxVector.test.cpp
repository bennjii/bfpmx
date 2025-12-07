//
// Test suite for MxVector
//

#define PROFILE 1
#include "profiler/profiler.h"

#include "definition/vector/MxVector.hpp"
#include "definition/prelude.h"
#include "helper/test.h"
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <string>

using TestingScalar = unsigned char;
using TestingFloat = fp8::E4M3Type;

template <typename Dimensions>
using TestingMxVector = mx::vector::MxVector<Dimensions, TestingScalar, TestingFloat,
                                              CPUArithmetic, MaximumFractionalQuantization>;

constexpr size_t TestingSize = 10000;
constexpr size_t TestIterations = 100;

using Vector = TestingMxVector<BlockDims<16>>;

TEST_CASE("MxVector Construction") {
  SECTION("Constructor from size") {
    Vector v(100);
    REQUIRE(v.Size() == 100);
    REQUIRE(v.NumBlocks() == (100 + 16 - 1) / 16);
  }

  SECTION("Constructor from std::vector<f64>") {
    std::vector<f64> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    Vector v(data);
    REQUIRE(v.Size() == 5);
    REQUIRE(v.NumBlocks() == (5 + 16 - 1) / 16);
  }

  SECTION("Constructor from blocks") {
    std::vector<Vector::BlockType> blocks;
    Vector v(blocks, 0);
    REQUIRE(v.Size() == 0);
    REQUIRE(v.NumBlocks() == 0);
  }
}

TEST_CASE("MxVector Element Access") {
  std::vector<f64> data = {1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5};
  Vector v(data);

  SECTION("ItemAt returns correct values") {
    for (size_t i = 0; i < data.size(); ++i) {
      REQUIRE(FuzzyEqual<TestingFloat>(v.ItemAt(i), data[i]));
    }
  }

  SECTION("BlockAt returns correct block") {
    const auto& block = v.BlockAt(0);
    REQUIRE(block.NumElems == 16);
  }
}

TEST_CASE("MxVector Element Modification") {
  Vector v(50);

  SECTION("SetItemAt modifies element") {
    bool success = v.SetItemAt(10, 42.0);
    REQUIRE(success);
    std::cout << v.ItemAt(10) << '\n';
    REQUIRE(FuzzyEqual<TestingFloat>(v.ItemAt(10), 42.0));
  }

  SECTION("SetItemAt out of bounds returns false") {
    bool success = v.SetItemAt(1000, 99.0);
    REQUIRE(!success);
  }

  SECTION("SetBlockAt modifies block") {
    Vector::BlockType new_block = Vector::BlockType::Quantize(
        std::array<f64, 16>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
    );
    bool success = v.SetBlockAt(0, new_block);
    REQUIRE(success);
  }

  SECTION("SetBlockAt out of bounds returns false") {
    Vector::BlockType new_block = Vector::BlockType::Quantize(
        std::array<f64, 16>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
    );
    bool success = v.SetBlockAt(100, new_block);
    REQUIRE(!success);
  }
}

TEST_CASE("MxVector Size Properties") {
  std::vector<f64> data(100);
  for (size_t i = 0; i < 100; ++i) {
    data[i] = i * 1.5;
  }
  Vector v(data);

  SECTION("Size returns correct count") {
    REQUIRE(v.Size() == 100);
  }

  SECTION("NumBlocks calculated correctly") {
    size_t expected_blocks = (100 + 16 - 1) / 16;
    REQUIRE(v.NumBlocks() == expected_blocks);
  }

  SECTION("SizeInBytes calculated correctly") {
    size_t expected_bytes = v.NumBlocks() * sizeof(Vector::BlockType);
    REQUIRE(v.SizeInBytes() == expected_bytes);
  }

  SECTION("NumBlockElements returns block size") {
    REQUIRE(v.NumBlockElements() == 16);
  }
}

// TEST_CASE("MxVector Quantization Accuracy") {
//   constexpr f64 min = 1.0, max = 100.0;
//   auto data = fill_random_arrays<f64, 64>(min, max);
//   Vector v(data);

//   SECTION("Quantized values match within tolerance") {
//     const f64 toleranceScaling = 1;
//     const i64 maxScalar = 255; // u8 max
//     const f64 epsilon = std::pow(2, maxScalar - static_cast<i64>(TestingFloat::SignificandBits()));

//     for (size_t i = 0; i < data.size(); ++i) {
//       f64 quantized = v.ItemAt(i);
//       f64 error = std::abs(quantized - data[i]);
//       REQUIRE(error <= epsilon * toleranceScaling);
//     }
//   }
// }

TEST_CASE("MxVector Edge Cases") {
  SECTION("Single element vector") {
    Vector v(1);
    REQUIRE(v.Size() == 1);
    REQUIRE(v.NumBlocks() == 1);
    
    v.SetItemAt(0, 42.5);
    REQUIRE(FuzzyEqual<TestingFloat>(v.ItemAt(0), 42.5));
  }

  SECTION("Large vector") {
    Vector v(10000);
    REQUIRE(v.Size() == 10000);
    REQUIRE(v.NumBlocks() == (10000 + 16 - 1) / 16);
  }

  SECTION("Vector with non-aligned size") {
    Vector v(17); // 17 is not a multiple of 16
    REQUIRE(v.Size() == 17);
    REQUIRE(v.NumBlocks() == 2);
  }
}

TEST_CASE("MxVector getBlocks") {
  std::vector<f64> data = {1, 2, 3, 4, 5};
  Vector v(data);

  SECTION("getBlocks returns correct number of blocks") {
    auto blocks = v.getBlocks();
    REQUIRE(blocks.size() == v.NumBlocks());
  }

  SECTION("Block data is preserved") {
    auto blocks = v.getBlocks();
    for (size_t i = 0; i < blocks.size(); ++i) {
      REQUIRE(blocks[i].NumElems == 16);
    }
  }
}

TEST_CASE("MxVector Independence") {
  std::vector<f64> data1 = {1, 2, 3, 4, 5};
  std::vector<f64> data2 = {10, 20, 30, 40, 50};

  Vector v1(data1);
  Vector v2(data2);

  SECTION("Vectors are independent") {
    REQUIRE(FuzzyEqual<TestingFloat>(v1.ItemAt(0), 1.0));
    REQUIRE(FuzzyEqual<TestingFloat>(v2.ItemAt(0), 10.0));

    v1.SetItemAt(0, 99.0);
    REQUIRE(FuzzyEqual<TestingFloat>(v1.ItemAt(0), 99.0));
    REQUIRE(FuzzyEqual<TestingFloat>(v2.ItemAt(0), 10.0)); // v2 unchanged
  }
}