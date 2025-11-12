#ifndef BFPMX_BLOCKDIMS_H
#define BFPMX_BLOCKDIMS_H
#include <array>
#include <cstddef>
#include <format>
#include <string>

#include "arch/prelude.h"
#include "definition/block_float/repr/FloatRepr.h"

template <u32... Dims> struct BlockDims {
  static constexpr u32 num_dims = sizeof...(Dims);
  using Dimensions = std::array<u32, num_dims>;
  static constexpr Dimensions values = {Dims...};

  static constexpr u32 TotalSize() {
    u32 prod = 1;
    for (auto d : values)
      prod *= d;
    return prod;
  }

  // Flatten coordinates → linear index
  static constexpr u32 CoordsToLinear(const Dimensions &coords) noexcept {
    u32 idx = 0;
    u32 stride = 1;
    for (std::size_t i = num_dims-1; i<num_dims; --i) {
      idx += coords[i] * stride;
      stride *= values[i];
    }
    return idx;
  }

  static constexpr Dimensions LinearToCoords(u32 linear) noexcept {
    Dimensions coords{};
    u32 remaining = linear;

    for (std::size_t i = 0; i < num_dims; ++i) {
      u32 stride = 1;
      for (std::size_t j = i + 1; j < num_dims; ++j) {
        stride *= values[j];
      }
      coords[i] = remaining / stride;
      remaining %= stride;
    }
    return coords;
  }
};

// Assure BlockShape is of type BlockDims
template <typename> struct isBlockDims : std::false_type {};

template <u32... Dims>
struct isBlockDims<BlockDims<Dims...>> : std::true_type {};

template <typename T>
concept BlockDimsType = isBlockDims<std::remove_cvref_t<T>>::value;
#endif