#include <iostream>

#include "definition/prelude.h"
using FloatReprCUDATest = fp8::E4M3Type;
template<template <typename T> typename ArithmeticPolicy>
using BlockT = Block<4, BlockDims<4>, FloatReprCUDATest, ArithmeticPolicy, MaximumFractionalQuantization>;
using BlockT_CPU = BlockT<CPUArithmetic>;
using BlockT_GPU = BlockT<GPUArithmetic>;

int main() {
    std::array<f64, 4> testArray1 = {10, 15, 20, 25};
    std::array<f64, 4> testArray2 = {2., 3., 4., 5.};
    // We are constructing a 4:3:1=8bit Float
    std::cout << "FloatSize=" << FloatReprCUDATest::Size() << " bits" << std::endl;

    // We pick particular arithmetic and quantization policies
    const auto block1_GPU = BlockT_GPU(testArray1);
    const auto block2_GPU = BlockT_GPU(testArray2);
    const auto block_GPU = block1_GPU + block2_GPU;

    const auto block1_CPU = BlockT_CPU(testArray1);
    const auto block2_CPU = BlockT_CPU(testArray2);
    const auto block_CPU = block1_CPU + block2_CPU;
    std::cout << "BlockGPU=" << block_GPU.asString() << std::endl;
    std::cout << "BlockCPU=" << block_CPU.asString() << std::endl;
    // Only returns correct value when run on devices with a NVIDIA GPU
    if (block_GPU.Spread() == block_CPU.Spread()) {
        std::cout << "GPU and CPU results match!" << std::endl;
    } else {
        std::cout << "Mismatch between GPU and CPU results!" << std::endl;
    }
}
