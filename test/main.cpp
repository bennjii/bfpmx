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
    using BlockT = Block<32, 4, FloatRepr, CPUArithmetic, MaximumFractionalQuantization>;

    const auto block1 = BlockT(std::to_array<f64, 4>({10, 15, 20, 25}));
    const auto block2 = BlockT(std::to_array<f64, 4>({2., 3., 4., 5.}));

    const auto block = block1 + block2;
    std::cout << "Block=" << block.asString() << std::endl;
}