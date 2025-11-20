//
// Created by Benjamin White on 13/10/2025.
//

#include <iostream>

#include "definition/prelude.h"

template <template <
    std::size_t, BlockDimsType, IFloatRepr,
    template <typename> typename ArithmeticPolicy_> typename QuantizationPolicy>
using TestingBlock = Block<4, BlockDims<32>, fp8::E4M3Type, CPUArithmetic,
                           QuantizationPolicy>;

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

    const auto block1 = TestingBlock<SharedExponentQuantization>::Quantize(arr);
    std::cout << "Quantized array {SEQ} : \n" << block1.asString() << std::endl;

    const auto block2 = TestingBlock<MaximumFractionalQuantization>::Quantize(arr);
    std::cout << "Quantized array {MFQ} : \n" << block2.asString() << std::endl;
}