//
// Created by Benjamin White on 24/10/2025.
//

#ifndef BFPMX_QUANTIZE_H
#define BFPMX_QUANTIZE_H
#include "arch/cpu/arithmetic.hpp"

template<
    template <std::size_t, std::size_t, IFloatRepr> typename T,
    std::size_t ScalarBytes,
    std::size_t Size,
    typename Float
>
concept IQuantize = IFloatRepr<Float> && requires(std::array<f64, Size> &v, Block<
        ScalarBytes,
        Size,
        Float,
        CPUArithmetic,
        T
    > &b) {
    { T<ScalarBytes, Size, Float>::Quantize(v) } -> std::convertible_to<Block<
        ScalarBytes,
        Size,
        Float,
        CPUArithmetic,
        T>
    >;
    { T<ScalarBytes, Size, Float>::UnQuantize(b) } -> std::same_as<std::array<f64, Size>>;
};

#endif //BFPMX_QUANTIZE_H