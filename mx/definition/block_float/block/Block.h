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

template<
    std::size_t ScalarSizeBytes, // TODO: document what is Scalar
    std::size_t BlockSizeElements,
    IFloatRepr Float,
    template<typename> typename ArithmeticPolicy,
    template<std::size_t, std::size_t, IFloatRepr> typename QuantizationPolicy
>
class Block : public
    WithPolicy<ArithmeticPolicy>::template
Type<Block<ScalarSizeBytes, BlockSizeElements, Float, ArithmeticPolicy, QuantizationPolicy>>
{
public:
    using FloatType = Float;
    using PackedFloat = std::array<u8, Float::SizeBytes()>;
    using ScalarType = std::array<u8, ScalarSizeBytes>;
    using QuantizationPolicyType = QuantizationPolicy<ScalarSizeBytes, BlockSizeElements, Float>;

    Block() {
        // TODO: We would need a quantisation layer that we can callout to
        //       ideally this is also pluggable, but could be a runtime dep
        //       instead of a static, typename, injection.

        auto data = std::array<PackedFloat, BlockSizeElements>();
             data.fill(Float::Marshal(0));

        data_ = data;

        auto scalar = std::array<u8, ScalarSizeBytes>();
             scalar.fill(0);

        scalar_ = scalar;
    }

    // TODO: Quantize
    Block(std::array<f64, BlockSizeElements> v) : Block(QuantizationPolicyType::Quantize(v)) {}

    // TODO: Quantize
    Block(std::array<f32, BlockSizeElements> v) {
        return;
    }

    Block(const Block&) = default;

    PackedFloat At(u16 index)
    {
        return data_[index];
    }

    // TODO: Quantize, and fill
    void Fill(f64 value)
    {
        return;
    }

    explicit Block(std::array<PackedFloat, BlockSizeElements> init) : data_(init), scalar_(0) {}
    Block(std::array<PackedFloat, BlockSizeElements> data, ScalarType scalar) : data_(data), scalar_(scalar) {}
    static constexpr std::size_t dataCount() { return BlockSizeElements; }

    PackedFloat* data() { return data_.data(); }
    [[nodiscard]] const PackedFloat* data() const { return data_.data(); }

    [[nodiscard]] static std::size_t Length()
    {
        return BlockSizeElements;
    }

    [[nodiscard]] u64 Scalar() const
    {
        u64 scalar = 0;
        for (int i = 0; i < ScalarSizeBytes; i++)
        {
            scalar |= scalar_[i] << (i * 8);
        }

        return scalar;
    }

    [[nodiscard]] std::string as_string() const {
        std::string value;

        value += "Scalar: " + std::to_string(Scalar()) + "\n";
        value += "Elements: [\n";
        for (PackedFloat v : data_) {
            auto unmarshalled = Float::Unmarshal(v);
            std::string valueUnscaled = std::format("{:.3f}", unmarshalled);
            std::string valueScaled = std::format("{:.3f}", unmarshalled * Scalar());

            value += "\t" + valueUnscaled + " (St. " + valueScaled + "), \n";
        }
        value += "]";

        return value;
    }

private:
    // Using Row-Major ordering
    std::array<PackedFloat, BlockSizeElements> data_;
    std::array<u8, ScalarSizeBytes> scalar_;
};

#endif //BFPMX_BLOCK_H
