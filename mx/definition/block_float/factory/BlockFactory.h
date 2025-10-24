//
// Created by Benjamin White on 10/10/2025.
//

#ifndef BFPMX_BLOCKFACTORY_H
#define BFPMX_BLOCKFACTORY_H

#include "definition/block_float/block/Block.h"

constexpr u16 DEFAULT_BLOCK_SIZE = 32;
constexpr u16 DEFAULT_BITS_SCALAR = 32;
constexpr u16 BITS_IN_BYTE = 8;

template<
    typename BlockShape,
    u16 BitsScalar,
    IFloatRepr Float,
    template<typename> typename ImplPolicy
>
class BlockFactory
{
public:
    BlockFactory() = delete;

    /// The size of the block, in bits, when constructed.
    static constexpr u32 Size() {
        return Float::Size() * BlockShape::total_size();
    }

    static constexpr Block<BitsScalar, BlockShape, Float, ImplPolicy> CreateBlock()
    {
        return Block<BitsScalar, BlockShape, Float, ImplPolicy>();
    }
};

#endif //BFPMX_BLOCKFACTORY_H