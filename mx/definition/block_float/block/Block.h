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

template<
    std::size_t ScalarSizeBytes,
    typename BlockShape,
    IFloatRepr Float,
    template<typename> typename P
>
class Block : public
    WithPolicy<P>::template Type<Block<ScalarSizeBytes, BlockShape, Float, P>>
{
public:
    static constexpr u32 num_dimensions = BlockShape::num_dims;
    static constexpr auto dims = BlockShape::values;
    static constexpr u32 num_elems = BlockShape::total_size();

    using PackedFloat = std::array<u8, Float::SizeBytes()>;

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

    // TODO: Quantize, and fill
    void Fill(f64 value)
    {
        return;
    }

    explicit Block(std::array<PackedFloat, ScalarSizeBytes> init) : data_(init), scalar_(0) {}

    [[nodiscard]] static std::size_t NumElems()
    {
        return num_elems;
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