//
// Created by Benjamin White on 10/10/2025.
//

#include <iostream>

#include "definition/prelude.h"

int main()
{
    // We are constructing a 4:3:1=8bit Float
    std::cout << "FloatSize=" << fp::E4M3Type::Size() << "bits" << std::endl;

    // We expect 16 + 32(8) = 272 bits
    using IBlock = BlockFactory<32, 16, fp::E4M3Type>;
    std::cout << "BlockSize=" << IBlock::Size() << "bits" << std::endl;

    // Below will be const-folded into Block<32, 272>::Length(), which statically
    // is known as 32.
    auto block = IBlock::CreateBlock();
    std::cout << "BlockLength=" << block.Length() << " elements" << std::endl;
}
