//
// Created by Benjamin White on 24/10/2025.
//

#ifndef BFPMX_QUANTIZE_H
#define BFPMX_QUANTIZE_H

template<
    typename T,
    typename Float,
    std::size_t ScalarBytes,
    std::size_t Size
>
concept IQuantize = IFloatRepr<Float> && requires(std::array<f64, Size> &v, Block<
        ScalarBytes,
        Size,
        Float,
        CPUArithmetic
    > &b) {
    { T::Quantize(v) } -> std::convertible_to<Block<
        ScalarBytes,
        Size,
        Float,
        CPUArithmetic
    >>;
    { T::UnQuantize(b) } -> std::same_as<std::array<f64, Size>>;
};

#endif //BFPMX_QUANTIZE_H