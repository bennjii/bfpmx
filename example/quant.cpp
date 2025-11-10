//
// Created by Benjamin White on 13/10/2025.
//

#include <iostream>

#include "definition/prelude.h"

int main() {
    std::array<f64, 32> arr = std::to_array<f64, 32>({
        1.2f, 3.4f, 5.6f, 2.1f,1.3f,-6.5f
    });

    // Example explicit quantization
    std::cout << "Starting array : [";
    for (std::size_t i = 0; i < arr.size(); i++) {
        std::cout << arr[i];

        if (i != arr.size() - 1)
        {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    const auto block1 = SharedExponentQuantization<4, BlockDims<32>, fp8::E5M2Type, CPUArithmetic>::Quantize(arr);
    std::cout << "Quantized array {SEQ} : \n" << block1.asString() << std::endl;

    const auto block2 = MaximumFractionalQuantization<4, BlockDims<32>, fp8::E4M3Type, CPUArithmetic>::Quantize(arr);
    std::cout << "Quantized array {MFQ} : \n" << block2.asString() << std::endl;
}