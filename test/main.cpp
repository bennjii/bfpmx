//
// Created by Benjamin White on 10/10/2025.
//

#include <iostream>

#include "definition/prelude.h"

int main() {
    // We are constructing a 4:3:1=8bit Float
    std::cout << "FloatSize=" << fp8::E4M3Type::Size() << "bits" << std::endl;

    // We expect 16 + 32(8) = 272 bits
    using BlockT = Block<32, 4, fp8::E4M3Type, CPUArithmetic, SharedExpQuantization>;

    // Below will be const-folded into Block<32, 272>::Length(), which statically
    // is known as 32.
    auto block1 = BlockT(std::to_array<f64, 4>({10, 15, 20, 25}));
    auto block2 = BlockT(std::to_array<f64, 4>({2., 3., 4., 5.}));
    const auto block = block2;
    std::cout << "BlockLength=" << block.Length() << " elements" << std::endl;
    std::cout << "block: " << block.as_string() << std::endl;
}