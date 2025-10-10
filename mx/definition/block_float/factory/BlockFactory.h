//
// Created by Benjamin White on 10/10/2025.
//

#ifndef BFPMX_BLOCKFACTORY_H
#define BFPMX_BLOCKFACTORY_H

#include "../repr/FloatRepr.h"

constexpr u16 DEFAULT_BLOCK_SIZE = 32;
constexpr u16 DEFAULT_BITS_SCALAR = 32;

class BlockFactory
{
public:
    BlockFactory() = delete;
    explicit BlockFactory(FloatRepr repr, u16 block_size, u16 scalar);

    static BlockFactory Default(const FloatRepr repr)
    {
        return BlockFactory(repr, DEFAULT_BLOCK_SIZE, DEFAULT_BITS_SCALAR);
    };

    void WithBlockSize(u16 size);
    void WithScalarBits(u16 scalar);

    /// The size of the block, in bits, when constructed.
    [[nodiscard]] u32 Size() const;

    // TODO: Block createBlock();

private:
    /// The representation of the floats within the block
    FloatRepr repr;

    /// The number of floats within a given block.
    /// Default is 32.
    u16 block_size;

    /// The size of the scalar value in bits.
    /// Default is 32.
    u16 bits_scalar;
};


#endif //BFPMX_BLOCKFACTORY_H