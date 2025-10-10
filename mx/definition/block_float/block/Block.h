//
// Created by Benjamin White on 10/10/2025.
//

#ifndef BFPMX_BLOCK_H
#define BFPMX_BLOCK_H
#include <array>


template<
    std::integral Scalar,
    std::size_t BlockSize
>
class Block
{
    std::array<Scalar, BlockSize> values;
};


#endif //BFPMX_BLOCK_H