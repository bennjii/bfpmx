//
// Created by Benjamin White on 10/10/2025.
//

#ifndef BFPMX_BLOCKFACTORY_H
#define BFPMX_BLOCKFACTORY_H

#include "../../prelude.h"
#include "../../../definition/quantization/Quantize.h"

constexpr u16 DEFAULT_BLOCK_SIZE = 32;
constexpr u16 DEFAULT_BITS_SCALAR = 32;
constexpr u16 BITS_IN_BYTE = 8;

template<
    u16 BlockQuantity,
    u16 BytesScalar,
    IFloatRepr Float,
    template<typename> typename ImplPolicy
>
class BlockFactory
{
public:
    BlockFactory() = delete;

    // Type alias for quantizers compatible with this factory's parameters
    template<template<std::size_t, std::size_t, typename> typename QuantizerTemplate>
    using BoundQuantizer = QuantizerTemplate<BytesScalar, BlockQuantity, Float>;

    /// The size of the block, in bits, when constructed.
    static constexpr u32 Size() {
        return Float::Size();
    }

    template<typename Quantizer>
        requires IQuantize<Quantizer, Float, BytesScalar, BlockQuantity>
    static constexpr Block<BytesScalar, BlockQuantity, Float, ImplPolicy> CreateBlock(Quantizer q)
    {
        return Block<BytesScalar, BlockQuantity, Float, ImplPolicy>();
    }
};

#endif //BFPMX_BLOCKFACTORY_H