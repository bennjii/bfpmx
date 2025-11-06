#include <iostream>

#include "definition/prelude.h"

int main() {
    using FloatRepr = fp8::E4M3Type;

    // We are constructing a 4:3:1=8bit Float
    std::cout << "FloatSize=" << FloatRepr::Size() << " bits" << std::endl;

    // We pick particular arithmetic and quantization policies
    using BlockT_GPU = Block<32, BlockDims<4>, FloatRepr, GPUArithmetic, MaximumFractionalQuantization>;
    const auto block1_GPU = BlockT_GPU(std::to_array<f64, 4>({10, 15, 20, 25}));
    const auto block2_GPU = BlockT_GPU(std::to_array<f64, 4>({2., 3., 4., 5.}));
    const auto block_GPU = block1_GPU + block2_GPU;
    using BlockT_CPU = Block<32, BlockDims<4>, FloatRepr, CPUArithmetic, MaximumFractionalQuantization>;
    const auto block1_CPU = BlockT_CPU(std::to_array<f64, 4>({10, 15, 20, 25}));
    const auto block2_CPU = BlockT_CPU(std::to_array<f64, 4>({2., 3., 4., 5.}));
    const auto block_CPU = block1_CPU + block2_CPU;
    std::cout << "BlockGPU=" << block_GPU.asString() << std::endl;
    std::cout << "BlockCPU=" << block_CPU.asString() << std::endl;
    if (block_GPU.Spread() == block_CPU.Spread()) {
        std::cout << "GPU and CPU results match!" << std::endl;
    } else {
        std::cout << "Mismatch between GPU and CPU results!" << std::endl;
    }
}
