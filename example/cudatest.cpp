#include <iostream>
#include <vector>

#include "definition/prelude.h"
#include "definition/vector/MxVector.hpp"
#include "definition/vector/MxVectorOperations.hpp"
#include "arch/gpu/preludeGPU.cuh"

using FloatReprCUDATest = fp8::E4M3Type;
using MxVectorTest = mx::vector::MxVector<BlockDims<32>, unsigned char, 
    FloatReprCUDATest, GPUArithmeticNaive, MaximumFractionalQuantization>;

int main() {
    std::cout << "=== MxVector GPU Test ===" << std::endl;
    std::cout << "FloatSize=" << FloatReprCUDATest::Size() << " bits" << std::endl;
    std::cout << "BlockSize=" << BlockDims<32>::TotalSize() << " elements" << std::endl;

    std::vector<f64> testData1 = {10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0,
                                   50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0};
    std::vector<f64> testData2 = {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                                   10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0};

    MxVectorTest mx1(testData1);
    MxVectorTest mx2(testData2);

    std::cout << "\n--- CPU Addition (Blockwise) ---" << std::endl;
    auto result_cpu = mx::vector::ops::AddBlockwise(mx1, mx2);
    std::cout << "First 4 elements: ";
    for (size_t i = 0; i < 4 && i < result_cpu.Size(); ++i) {
        std::cout << result_cpu.ItemAt(i) << " ";
    }
    std::cout << std::endl;

    std::cout << "\n--- GPU Addition (Fused) ---" << std::endl;
    auto result_gpu = mx::vector::ops::AddPointwiseGPUFused(mx1, mx2);
    std::cout << "  (Result is automatically converted back to CPU MxVector)" << std::endl;
    
    std::cout << "\n--- Accessing GPU result ---" << std::endl;
    std::cout << "First 4 elements: ";
    for (size_t i = 0; i < 4 && i < result_gpu.Size(); ++i) {
        std::cout << result_gpu.ItemAt(i) << " ";
    }
    std::cout << std::endl;

    std::cout << "\n--- Verification ---" << std::endl;
    bool match = true;
    const double tolerance = 1e-6;
    for (size_t i = 0; i < std::min(result_cpu.Size(), result_gpu.Size()); ++i) {
        double diff = std::abs(result_cpu.ItemAt(i) - result_gpu.ItemAt(i));
        if (diff > tolerance) {
            match = false;
            std::cout << "Mismatch at index " << i << ": CPU=" << result_cpu.ItemAt(i) 
                      << ", GPU=" << result_gpu.ItemAt(i) << std::endl;
            break;
        }
    }
    
    if (match) {
        std::cout << "GPU and CPU results match!" << std::endl;
    } else {
        std::cout << "Mismatch between GPU and CPU results!" << std::endl;
    }

    std::cout << "\n--- Testing Second GPU Operation ---" << std::endl;
    auto result_gpu2 = mx::vector::ops::AddPointwiseGPUFused(result_gpu, mx1);
    std::cout << "Second GPU operation completed (result converted back to CPU)" << std::endl;

    return match ? 0 : 1;
}
