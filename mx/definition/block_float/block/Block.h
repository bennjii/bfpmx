//
// Created by Benjamin White on 10/10/2025.
//

#ifndef BFPMX_BLOCK_H
#define BFPMX_BLOCK_H

#include <array>
#include "definition/prelude.h"

template<
    std::size_t BlockQuantity,
    std::size_t BlockSize
>
class Block
{
public:
    Block() = default;

    static constexpr u32 Length()
    {
        return BlockQuantity;
    }

private:
    std::array<u8, BlockSize> values;
    u32 quantity = BlockQuantity;
};

#endif //BFPMX_BLOCK_H