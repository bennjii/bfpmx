//
// Created by Benjamin White on 10/10/2025.
//

#include "BlockFactory.h"

#include <iostream>

BlockFactory::BlockFactory(const FloatRepr repr, const u16 block_size, const u16 scalar) : repr(repr)
{
    this->repr = repr;
    this->block_size = block_size;
    this->bits_scalar = scalar;
}

void BlockFactory::WithBlockSize(const u16 size)
{
    this->block_size = size;
}

void BlockFactory::WithScalarBits(const u16 scalar)
{
    this->bits_scalar = scalar;
}

u32 BlockFactory::Size() const
{
    return this->bits_scalar + (this->block_size * this->repr.Size());
}