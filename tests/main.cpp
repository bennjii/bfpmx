//
// Created by Benjamin White on 10/10/2025.
//

#include <iostream>

#include "definition/prelude.h"

int main()
{
    // Using the E4M3 (FP8 repr)
    const auto repr = FloatRepr::E4M3();
    BlockFactory factory = BlockFactory::Default(repr);

    // 32 Blocks
    factory.WithBlockSize(32);
    // Using 16 scalar bits
    factory.WithScalarBits(16);

    // We expect 16 + 32(8) = 272 bits
    std::cout << "ConstructedSize=" << factory.Size();
}
