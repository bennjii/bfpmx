//
// Created by Benjamin White on 10/10/2025.
//

#include <iostream>

#include "definition/prelude.h"

int main() {
    using FloatRepr = fp8::E4M3Type;

    // We are constructing a 4:3:1=8bit Float
    std::cout << "FloatSize=" << FloatRepr::Size() << " bits" << std::endl;

    // We pick particular arithmetic and quantization policies
    using BlockT = Block<u32, BlockDims<4>, FloatRepr, CPUArithmetic, MaximumFractionalQuantization>;

    const auto block1 = BlockT(std::to_array<f64, 4>({10, 15, 20, 25}));
    const auto block2 = BlockT(std::to_array<f64, 4>({2., 3., 4., 5.}));

    const auto block = block1 + block2;
    std::cout << "Block=" << block.asString() << std::endl;

    // We expect 16 + 32(8) = 272 bits
    using IBlock = BlockFactory<BlockDims<2, 2, 2>, u32, fp8::E4M3Type, CPUArithmetic, MaximumFractionalQuantization>;
    std::cout << "BlockSize=" << IBlock::Size() << "bits" << std::endl;
    std::cout << "BlockSize=" << IBlock::Size() / BITS_IN_BYTE << "bytes" << std::endl;

    // Below will be const-folded into Block<4, 272>::Length(), which statically
    // is known as 32.
    const auto block3 = IBlock::CreateBlock();
    std::cout << "BlockNumElems=" << block3.NumElems << " elements" << std::endl;

    std::cout << "block3: " << block3.asString() << std::endl;
    auto newBlock = block3 + block3;
    newBlock(0, 0, 0) = fp8::E4M3Type::Marshal(1);
    newBlock(1, 1, 1) = fp8::E4M3Type::Marshal(2);
    std::cout << "new block: " << newBlock.asString() << std::endl;
}
