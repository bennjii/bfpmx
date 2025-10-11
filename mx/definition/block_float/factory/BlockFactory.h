//
// Created by Benjamin White on 10/10/2025.
//

#ifndef BFPMX_BLOCKFACTORY_H
#define BFPMX_BLOCKFACTORY_H

#include "definition/prelude.h"

constexpr u16 DEFAULT_BLOCK_SIZE = 32;
constexpr u16 DEFAULT_BITS_SCALAR = 32;

template<typename T>
concept IFloatRepr = requires(const T& s) {
    { s.Size() } -> std::convertible_to<u16>;
};

template<
    std::size_t BlockQuantity,
    std::size_t BlockSize,
    template<typename> typename ImplPolicy
>
class Block;

template<
    u16 BlockQuantity,
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
        return BitsScalar + (BlockQuantity * Float::Size());
    }

    static constexpr Block<BlockQuantity, Size(), ImplPolicy> CreateBlock()
    {
        return Block<BlockQuantity, Size(), ImplPolicy>();
    }
};

#endif //BFPMX_BLOCKFACTORY_H