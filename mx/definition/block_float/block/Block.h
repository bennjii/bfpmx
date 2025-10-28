//
// Created by Benjamin White on 10/10/2025.
//

#ifndef BFPMX_BLOCK_H
#define BFPMX_BLOCK_H

#include <array>
#include <cstddef>
#include <format>
#include <string>

#include "arch/prelude.h"
#include "definition/block_float/repr/FloatRepr.h"

// Hook to simplify type definitions for wrapping a class
// as supporting the custom prescribed arithmetic.
template<template<typename> typename ImplPolicy>
struct WithPolicy {
    template<typename T>
    using Type = ArithmeticEnabled<T, ImplPolicy<T>>;
};

template<u32... Dims>
struct BlockDims {
    static constexpr u32 num_dims = sizeof...(Dims);
    static constexpr std::array<u32, num_dims> values = {Dims...};

    static constexpr u32 total_size() {
        u32 prod = 1;
        for (auto d : values) prod *= d;
        return prod;
    }

    // Flatten coordinates → linear index
    static constexpr u32 coords_to_linear(const std::array<u32, num_dims>& coords) noexcept {
        u32 idx = 0;
        u32 stride = 1;
        for (std::size_t i = 0; i < num_dims; ++i) {
            idx += coords[i] * stride;
            stride *= values[i];
        }
        return idx;
    }
};

// Assure BlockShape is of type BlockDims
template<typename>
struct is_block_dims : std::false_type {};

template<u32... Dims>
struct is_block_dims<BlockDims<Dims...>> : std::true_type {};

template<typename T>
concept BlockDimsType = is_block_dims<std::remove_cvref_t<T>>::value;

template<
    std::size_t ScalarSizeBytes,
    BlockDimsType BlockShape,
    IFloatRepr Float,
    template<typename> typename ArithmeticPolicy,
    template<std::size_t, BlockDimsType, IFloatRepr> typename QuantizationPolicy
>
class Block : public
    WithPolicy<ArithmeticPolicy>::template
Type<Block<ScalarSizeBytes, BlockShape, Float, ArithmeticPolicy, QuantizationPolicy>>
{
public:
    using FloatType = Float;

    static constexpr u32 num_dimensions = BlockShape::num_dims;
    static constexpr auto dims = BlockShape::values;
    static constexpr u32 num_elems = BlockShape::total_size();

    using PackedFloat = std::array<u8, Float::SizeBytes()>;
    using ScalarType = std::array<u8, ScalarSizeBytes>;
    using QuantizationPolicyType = QuantizationPolicy<ScalarSizeBytes, BlockShape, Float>;

    // Empty constructor
    Block() {
        // TODO: We would need a quantisation layer that we can callout to
        //       ideally this is also pluggable, but could be a runtime dep
        //       instead of a static, typename, injection.

        auto data = std::array<PackedFloat, num_elems>();
             data.fill(Float::Marshal(0));

        data_ = data;

        auto scalar = std::array<u8, ScalarSizeBytes>();
             scalar.fill(0);

        scalar_ = scalar;
    }

    // Constructors from given element types
    explicit Block(std::array<f64, num_elems> v) : Block(QuantizationPolicyType::Quantize(v)) {}
    explicit Block(std::array<PackedFloat, num_elems> init) : data_(init), scalar_(0) {}
    explicit Block(std::array<PackedFloat, num_elems> data, ScalarType scalar) : data_(data), scalar_(scalar) {}

    Block(const Block&) = default;

    [[nodiscard]] static constexpr std::size_t Length()
    {
        return num_elems;
    }

    [[nodiscard]] PackedFloat At(u16 index) const
    {
        return data_[index];
    }

    explicit Block(std::array<PackedFloat, ScalarSizeBytes> init) : data_(init), scalar_(0) {}

    [[nodiscard]] static std::size_t NumElems()
    {
        return num_elems;
    }

    [[nodiscard]] u64 Scalar() const
    {
        u64 scalar = 0;
        for (int i = 0; i < ScalarSizeBytes; i++)
        {
            scalar |= scalar_[i] << (i * 8);
        }

        return 1 << scalar;
    }

    [[nodiscard]] std::array<f64, num_elems> Spread() const
    {
        std::array<f64, num_elems> blockUnscaledFloats;
        for (int i = 0; i < num_elems; i++)
        {
            auto packedFloat = At(i);
            const f64 fullPrecision = Float::Unmarshal(packedFloat);
            blockUnscaledFloats[i] = fullPrecision * Scalar();
        }

        return blockUnscaledFloats;
    }

    [[nodiscard]] std::string asString() const {
        std::array<f64, num_elems> fullPrecisionValues = Spread();
        std::string value;

        value += "Scalar: " + std::to_string(Scalar()) + "\n";
        value += "Elements: [\n";
        for (int i = 0; i < num_elems; i++)
        {
            f64 fullPrecisionFloat = fullPrecisionValues[i];
            value += std::format("\t ({}) {:.3f} \n", i, fullPrecisionFloat);
        }
        value += "]";

        return value;
    }
    
    // Templated for parameter packs
    template<typename... IndexTypes>
    constexpr PackedFloat& operator()(IndexTypes... idxs) noexcept {
        static_assert(sizeof...(idxs) == BlockShape::num_dims,
                      "Incorrect number of indices for this Block");
        std::array<u32, sizeof...(idxs)> coords{static_cast<u32>(idxs)...};
        u32 linear = BlockShape::coords_to_linear(coords);
        return data_[linear];
    }

    // Templated for parameter packs
    template<typename... IndexTypes>
    constexpr const PackedFloat& operator()(IndexTypes... idxs) const noexcept {
        static_assert(sizeof...(idxs) == BlockShape::num_dims,
                      "Incorrect number of indices for this Block");
        std::array<u32, sizeof...(idxs)> coords{static_cast<u32>(idxs)...};
        u32 linear = BlockShape::coords_to_linear(coords);
        return data_[linear];
    }

private:
    // Using Row-Major ordering
    std::array<PackedFloat, num_elems> data_;
    std::array<u8, ScalarSizeBytes> scalar_;
};

#endif //BFPMX_BLOCK_H
