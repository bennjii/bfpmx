//
// Created by Benjamin White on 10/10/2025.
//

#include <iostream>

#include "definition/prelude.h"

int main()
{
    // We are constructing a 4:3:1=8bit Float
    std::cout << "FloatSize=" << fp8::E4M3Type::Size() << "bits" << std::endl;

    // We expect 16 + 32(8) = 272 bits
    using IBlock = BlockFactory<BlockDims<2, 2, 2>, 16, fp8::E4M3Type, CPUArithmetic>;
    std::cout << "BlockSize=" << IBlock::Size() << "bits" << std::endl;
    std::cout << "BlockSize=" << IBlock::Size() / BITS_IN_BYTE << "bytes" << std::endl;

    // Below will be const-folded into Block<32, 272>::Length(), which statically
    // is known as 32.
    const auto block = IBlock::CreateBlock();
    std::cout << "BlockNumElems=" << block.NumElems() << " elements" << std::endl;

    std::cout << "block: " << block.as_string() << std::endl;
    auto newBlock = block + block;
    newBlock(0, 0, 0) = fp8::E4M3Type::Marshal(1);
    newBlock(1, 1, 1) = fp8::E4M3Type::Marshal(2);
    std::cout << "new block: " << newBlock.as_string() << std::endl;
}
