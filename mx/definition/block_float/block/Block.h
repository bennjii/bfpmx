//
// Created by Benjamin White on 10/10/2025.
//

#ifndef BFPMX_BLOCK_H
#define BFPMX_BLOCK_H

#include <array>
#include <cstddef>
#if !defined(__CUDACC__)
    // Pure host (g++, clang++)
    #include <format>
#endif
#include <cstring>
#include <string>

#include "BlockDims.h"
#include "arch/prelude.h"
#include "definition/block_float/repr/FloatRepr.h"

template <template <typename> typename ImplPolicy> struct WithPolicy {
  template <typename T> using Type = ArithmeticEnabled<T, ImplPolicy<T>>;
};

template <
    // The unsigned integer to use as the scalar
    typename Scalar,
    // The dimensionality of the block with regard to shape
    BlockDimsType BlockShape,
    // The representation of the floating point values within the block
    IFloatRepr Float,
    // The arithmetic policy to use for mathematical functions
    template <typename> typename ArithmeticPolicy,
    // The quantization policy to use
    template <std::size_t, BlockDimsType,
              IFloatRepr> typename QuantizationPolicy>
class Block
    : public WithPolicy<ArithmeticPolicy>::template Type<Block<
          Scalar, BlockShape, Float, ArithmeticPolicy, QuantizationPolicy>> {
  // Statically confirm the provided scalar is an unsigned integer value
  static_assert(std::is_integral_v<Scalar> && std::is_unsigned_v<Scalar>,
                "Template parameter T must be an unsigned integer.");

public:
  using FloatType = Float;
  using ScalarType = Scalar;
  using Shape = BlockShape;

  static constexpr size_t ScalarSizeBytes = sizeof(Scalar);

  static constexpr u32 NumDimensions = BlockShape::num_dims;
  static constexpr auto Dims = BlockShape::values;
  static constexpr u32 NumElems = BlockShape::TotalSize();
  static constexpr u32 ScalarBytes = ScalarSizeBytes;


  using PackedFloat = std::array<u8, Float::SizeBytes()>;
  using QuantizationPolicyType =
      QuantizationPolicy<ScalarSizeBytes, BlockShape, Float>;

  // Empty constructor
  explicit Block() {
    auto data = std::array<PackedFloat, NumElems>();
    data.fill(Float::Marshal(0));

    data_ = data;
    scalar_ = 0;
  }

  struct Uninitialized {};
  static inline constexpr Uninitialized UninitializedUnit{};
  // Usage: `Block b{Block::Uninitialized};`
  explicit Block(Uninitialized) noexcept {
    // do not initialize data_
  }

  // Constructors from given element types
  explicit Block(std::array<f64, NumElems> v) : Block(Quantize(v)) {}
  explicit Block(std::array<PackedFloat, NumElems> init)
      : data_(init), scalar_(0) {}
  explicit Block(std::array<PackedFloat, NumElems> data, Scalar scalar)
      : data_(data), scalar_(scalar) {}

  Block(const Block &) = default;

  [[nodiscard]] static constexpr std::size_t Length() { return NumElems; }

  [[nodiscard]] static constexpr std::size_t SizeBytes() {
    return NumElems * Float::SizeBytes() + ScalarSizeBytes;
  }

  [[nodiscard]] std::optional<PackedFloat> At(const u16 index) const {
    if (index >= NumElems) {
      return std::nullopt;
    }

    return AtUnsafe(index);
  }

  void SetValue(const u16 index, f64 value) {
    SetPackedBitsAtUnsafe(index, Float::Marshal(value / ScalarValue()));
  }

  // A variant of `At` which runs on the provided assertions that
  // the underlying data must exist at the index.
  HD [[nodiscard]] PackedFloat AtUnsafe(const u16 index) const {
    return data_[index];
  }

  template <typename T> HD [[nodiscard]] T AtUnsafeBits(const u16 index) const {
    auto d = data_[index];
    T bits;
    // memcpy is optimized at comp-time since aPacked.size() is a constexpr
    std::memcpy(&bits, d.data(), d.size());
    return bits;
  }

  void SetPackedBitsAtUnsafe(const u16 index,
                             std::array<u8, Float::SizeBytes()> const &bits) {
    data_[index] = bits;
  }

  void SetBitsAtUnsafe(const u16 index, const u64 bits) {
    std::array<u8, Float::SizeBytes()> out{};
    // memcpy is optimized at comp-time since Float::SizeBytes() is a constexpr
    std::memcpy(&out[0], &bits, Float::SizeBytes());
    SetPackedBitsAtUnsafe(index, out);
  }

  [[nodiscard]] std::optional<f64> RealizeAt(const u16 index) const {
    if (index >= NumElems) {
      return std::nullopt;
    }

    return RealizeAtUnsafe(index);
  }

  HD [[nodiscard]] f64 RealizeAtUnsafe(const u16 index) const {
    return Float::Unmarshal(AtUnsafe(index)) * ScalarValue();
  }

  void SetScalar(Scalar scalar) { scalar_ = scalar; }

  [[nodiscard]] constexpr Scalar inline ScalarBits() const { return scalar_; }
  [[nodiscard]] constexpr Scalar inline ScalarValue() const {
    return 1 << scalar_;
  }

  f64 ElementAt(const u16 index) const {
    if (index >= NumElems) {
      throw std::out_of_range("Index out of range in Block::ElementAt");
    }
    return Float::Unmarshal(AtUnsafe(index));
  }

  [[nodiscard]] std::array<f64, NumElems> Spread() const {
    std::array<f64, NumElems> blockUnscaledFloats;
    for (int i = 0; i < NumElems; i++) {
      blockUnscaledFloats[i] = RealizeAtUnsafe(i);
    }

    return blockUnscaledFloats;
  }

  // TODO: Heuristic to rescale to block, for now no requantization
  void SetItemAt(const u16 index, f64 value) {
    if (index >= NumElems) {
      throw std::out_of_range("Index out of range in Block::SetItemAt");
    }
    data_.at(index) = Float::Marshal(value / Scalar());
  }

  // TODO: Heuristic to rescale to block, for now no requantization
  void SetItemAtUnsafe(const u16 index, f64 value) {
    data_.at(index) = Float::Marshal(value / Scalar());
  }

  [[nodiscard]] std::string asString() const {
    // TODO: Nested square brackets for multidim blocks
    std::array<f64, NumElems> fullPrecisionValues = Spread();
    std::string value;

    value += "Scalar: " + std::to_string(ScalarValue()) + "\n";
    value += "Elements: [\n";
    for (int i = 0; i < NumElems; i++) {
      f64 fullPrecisionFloat = fullPrecisionValues[i];
      std::string end = (i == NumElems - 1) ? "" : ", ";
      value += std::format("({}) {:.3f}{}", i, fullPrecisionFloat, end);
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
  constexpr const f64 &operator[](IndexTypes... idxs) const noexcept {
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
    return Block(blockScaledFloats, scaleFactorInt);
  }

  auto data() const { return data_; }
  auto scalar() const { return scalar_; }

private:
  // Using Row-Major ordering
  std::array<PackedFloat, NumElems> data_;
  Scalar scalar_;
};

#endif // BFPMX_BLOCK_H
