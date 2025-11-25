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

template <
    std::size_t ScalarSizeBytes, BlockDimsType BlockShape, IFloatRepr Float,
    template <typename> typename ArithmeticPolicy,
    template <
        std::size_t, BlockDimsType, IFloatRepr,
        template <
            typename> typename ArithmeticPolicy_> typename QuantizationPolicy>
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
      QuantizationPolicy<ScalarSizeBytes, BlockShape, Float, ArithmeticPolicy>;

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

  struct _Uninitialized {};
  static inline constexpr _Uninitialized Uninitialized{};
  // Usage: `Block b{Block::Uninitialized};`
  Block(_Uninitialized) noexcept {
    // do not initialize data_
  }

  // Constructors from given element types
  explicit Block(std::array<f64, NumElems> v) : Block(Quantize(v)) {}
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

  void SetValue(const u16 index, f64 value) const {
    return SetBitsAtUnsafe(index, Float::Marshal(value));
  }

  // A variant of `At` which runs on the provided assertions that
  // the underlying data must exist at the index.
  [[nodiscard]] PackedFloat AtUnsafe(const u16 index) const {
    return data_[index];
  }

  void SetPackedBitsAtUnsafe(const u16 index,
                             std::array<u8, Float::SizeBytes()> const &bits) {
    data_[index] = bits;
  }

  void SetBitsAtUnsafe(const u16 index, const u64 bits) {
    // TODO: see ScalarBits
    std::array<u8, Float::SizeBytes()> out{};
    for (size_t i = 0; i < Float::SizeBytes(); ++i) {
      out[i] = static_cast<u8>(bits >> (i * 8));
    }
    SetPackedBitsAtUnsafe(index, out);
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

  void SetScalar(u64 scalar) {
    // TODO: see ScalarBits
    for (int i = 0; i < ScalarSizeBytes; i++) {
      scalar_[i] = scalar >> (i * 8);
    }
  }

  [[nodiscard]] u64 ScalarBits() const {
    // TODO: can this be optimized with a simple cast,
    //       like `if (ScalarSizeBytes == 1) return *(*u8) scalar_;`
    //            `if (ScalarSizeBytes == 2) return *(*u16) scalar_;`
    //       or even better: instead of specifying `ScalarSizeBytes` cannot we
    //       specify a type, like u8, u16, u32 or u64?
    u64 scalar = 0;
    for (int i = 0; i < ScalarSizeBytes; i++) {
      scalar |= scalar_[i] << (i * 8);
    }
    return scalar;
  }

  [[nodiscard]] u64 Scalar() const { return 1 << ScalarBits(); }

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
  constexpr f64 operator[](IndexTypes... idxs) noexcept {
    static_assert(sizeof...(idxs) == BlockShape::num_dims,
                  "Incorrect number of indices for this Block");
    std::array<u32, sizeof...(idxs)> coords{static_cast<u32>(idxs)...};
    const u32 linear = BlockShape::CoordsToLinear(coords);
    return RealizeAtUnsafe(linear);
  }

  // Templated for parameter packs
  template <typename... IndexTypes>
  constexpr const f64 &
  operator[](IndexTypes... idxs) const noexcept {
    static_assert(sizeof...(idxs) == BlockShape::num_dims,
                  "Incorrect number of indices for this Block");
    std::array<u32, sizeof...(idxs)> coords{static_cast<u32>(idxs)...};
    const u32 linear = BlockShape::CoordsToLinear(coords);
    return RealizeAt(linear);
  }

  static Block Quantize(const std::array<f64, BlockShape::TotalSize()> &vec) {
    f64 scaleFactor = QuantizationPolicyType::QuantizerScaleFactor(vec);

    // Scale each element to become x_i = v_i / S.
    std::array<PackedFloat, BlockShape::TotalSize()> blockScaledFloats;
    for (int i = 0; i < BlockShape::TotalSize(); i++) {
      f64 scaledValue = vec[i] / scaleFactor;
      blockScaledFloats[i] = Float::Marshal(scaledValue);
    }

    const u32 scaleFactorInt = lround(log2(scaleFactor));

    std::array<u8, ScalarSizeBytes> packedScalar;
    for (int i = 0; i < ScalarSizeBytes; i++) {
      packedScalar[i] = static_cast<u8>(scaleFactorInt >> (i * 8));
    }

    return Block(blockScaledFloats, packedScalar);
  }

private:
  // Using Row-Major ordering
  std::array<PackedFloat, NumElems> data_;
  std::array<u8, ScalarSizeBytes> scalar_;
};

#endif // BFPMX_BLOCK_H
