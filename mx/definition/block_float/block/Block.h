//
// Created by Benjamin White on 10/10/2025.
//

#ifndef BFPMX_BLOCK_H
#define BFPMX_BLOCK_H

#include <array>
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
    std::size_t ScalarSizeBytes,
    std::size_t BlockSizeElements,
    IFloatRepr Float,
    template<typename> typename P
>
class Block : public
    WithPolicy<P>::template Type<Block<ScalarSizeBytes, BlockSizeElements, Float, P>>
{
public:
    using PackedFloat = std::array<u8, Float::SizeBytes()>;

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

    // TODO: Quantize, and fill
    void Fill(f64 value)
    {
        return;
    }

    explicit Block(std::array<PackedFloat, ScalarSizeBytes> init) : data_(init), scalar_(0) {}

    [[nodiscard]] static std::size_t Length()
    {
        return Float::Size();
    }

    PackedFloat* data() { return data_.data(); }
    [[nodiscard]] const PackedFloat* data() const { return data_.data(); }

    [[nodiscard]] std::string as_string() const {
        std::string value;
        value += "[";
        for (PackedFloat v : data_) {
            std::string s = std::format("{:.3f}", Float::Unmarshal(v));
            value += s + ", ";
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