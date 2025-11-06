//
// Created by Benjamin White on 10/10/2025.
//

#ifndef BFPMX_BLOCK_H
#define BFPMX_BLOCK_H

#include <array>
#include <cstddef>
#include <format>
#include <string>

#include "BlockDims.h"
#include "arch/prelude.h"
#include "definition/block_float/repr/FloatRepr.h"

// Hook to simplify type definitions for wrapping a class
// as supporting the custom prescribed arithmetic.
template <template <typename> typename ImplPolicy> struct WithPolicy {
  template <typename T> using Type = ArithmeticEnabled<T, ImplPolicy<T>>;
};

template <std::size_t ScalarSizeBytes, BlockDimsType BlockShape,
          IFloatRepr Float, template <typename> typename ArithmeticPolicy,
          template <std::size_t, BlockDimsType,
                    IFloatRepr> typename QuantizationPolicy>
class Block : public WithPolicy<ArithmeticPolicy>::template Type<
                  Block<ScalarSizeBytes, BlockShape, Float, ArithmeticPolicy,
                        QuantizationPolicy>> {
public:
  using FloatType = Float;

  static constexpr u32 NumDimensions = BlockShape::num_dims;
  static constexpr auto Dims = BlockShape::values;
  static constexpr u32 NumElems = BlockShape::TotalSize();

  using PackedFloat = std::array<u8, Float::SizeBytes()>;
  using ScalarType = std::array<u8, ScalarSizeBytes>;
  using QuantizationPolicyType =
      QuantizationPolicy<ScalarSizeBytes, BlockShape, Float>;

  // Empty constructor
  Block() {
    // TODO: We would need a quantisation layer that we can callout to
    //       ideally this is also pluggable, but could be a runtime dep
    //       instead of a static, typename, injection.

    auto data = std::array<PackedFloat, NumElems>();
    data.fill(Float::Marshal(0));

    data_ = data;

    auto scalar = std::array<u8, ScalarSizeBytes>();
    scalar.fill(0);

    scalar_ = scalar;
  }

  // Constructors from given element types
  explicit Block(std::array<f64, NumElems> v)
      : Block(QuantizationPolicyType::Quantize(v)) {}
  explicit Block(std::array<PackedFloat, NumElems> init)
      : data_(init), scalar_(0) {}
  explicit Block(std::array<PackedFloat, NumElems> data, ScalarType scalar)
      : data_(data), scalar_(scalar) {}

  Block(const Block &) = default;

  [[nodiscard]] static constexpr std::size_t Length() { return NumElems; }

  [[nodiscard]] std::optional<PackedFloat> At(const u16 index) const {
    if (index >= NumElems) {
      return std::nullopt;
    }

    return AtUnsafe(index);
  }

  // A variant of `At` which runs on the provided assertions that
  // the underlying data must exist at the index.
  [[nodiscard]] PackedFloat AtUnsafe(const u16 index) const {
    return data_[index];
  }

  [[nodiscard]] std::optional<f64> RealizeAt(const u16 index) const {
    if (index >= NumElems) {
      return std::nullopt;
    }

    return RealizeAtUnsafe(index);
  }

  [[nodiscard]] f64 RealizeAtUnsafe(const u16 index) const {
    return Float::Unmarshal(AtUnsafe(index)) * Scalar();
  }

  [[nodiscard]] u64 Scalar() const {
    u64 scalar = 0;
    for (int i = 0; i < ScalarSizeBytes; i++) {
      scalar |= scalar_[i] << (i * 8);
    }

    return 1 << scalar;
  }

  [[nodiscard]] std::array<f64, NumElems> Spread() const {
    std::array<f64, NumElems> blockUnscaledFloats;
    for (int i = 0; i < NumElems; i++) {
      blockUnscaledFloats[i] = RealizeAtUnsafe(i);
    }

    return blockUnscaledFloats;
  }

  [[nodiscard]] std::string asString() const {
    // TODO: Nested square brackets for multidim blocks
    std::array<f64, NumElems> fullPrecisionValues = Spread();
    std::string value;

    value += "Scalar: " + std::to_string(Scalar()) + "\n";
    value += "Elements: [\n";
    for (int i = 0; i < NumElems; i++) {
      f64 fullPrecisionFloat = fullPrecisionValues[i];
      value += std::format("\t ({}) {:.3f} \n", i, fullPrecisionFloat);
    }
    value += "]";

    return value;
  }

  // Templated for parameter packs
  template <typename... IndexTypes>
  constexpr PackedFloat &operator()(IndexTypes... idxs) noexcept {
    static_assert(sizeof...(idxs) == BlockShape::num_dims,
                  "Incorrect number of indices for this Block");
    std::array<u32, sizeof...(idxs)> coords{static_cast<u32>(idxs)...};
    u32 linear = BlockShape::CoordsToLinear(coords);
    return data_.at(linear);
  }

  // Templated for parameter packs
  template <typename... IndexTypes>
  constexpr const PackedFloat &operator()(IndexTypes... idxs) const noexcept {
    static_assert(sizeof...(idxs) == BlockShape::num_dims,
                  "Incorrect number of indices for this Block");
    std::array<u32, sizeof...(idxs)> coords{static_cast<u32>(idxs)...};
    u32 linear = BlockShape::CoordsToLinear(coords);
    return data_.at(linear);
  }

  // Templated for parameter packs
  template <typename... IndexTypes>
  constexpr std::optional<f64> operator[](IndexTypes... idxs) noexcept {
    static_assert(sizeof...(idxs) == BlockShape::num_dims,
                  "Incorrect number of indices for this Block");
    std::array<u32, sizeof...(idxs)> coords{static_cast<u32>(idxs)...};
    const u32 linear = BlockShape::CoordsToLinear(coords);
    return RealizeAt(linear);
  }

  // Templated for parameter packs
  template <typename... IndexTypes>
  constexpr const std::optional<f64> &
  operator[](IndexTypes... idxs) const noexcept {
    static_assert(sizeof...(idxs) == BlockShape::num_dims,
                  "Incorrect number of indices for this Block");
    std::array<u32, sizeof...(idxs)> coords{static_cast<u32>(idxs)...};
    const u32 linear = BlockShape::CoordsToLinear(coords);
    return RealizeAt(linear);
  }

  static Block Quantize(const std::array<f64, BlockShape::TotalSize()> &vec) {
    return QuantizationPolicyType::Quantize(vec);
  }

private:
  // Using Row-Major ordering
  std::array<PackedFloat, NumElems> data_;
  std::array<u8, ScalarSizeBytes> scalar_;
};

#endif // BFPMX_BLOCK_H
