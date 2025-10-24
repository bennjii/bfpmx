//
// Created by Benjamin White on 13/10/2025.
//

#include <iostream>

#include "definition/prelude.h"

int main() {
    std::array<f64, 32> arr = std::to_array<f64, 32>({
        1.2f, 3.4f, 5.6f, 2.1f,1.3f,-6.5f
    });

<<<<<<< HEAD
    std::cout << "Starting array : [";
    for (std::size_t i = 0; i < arr.size(); i++) {
        std::cout << arr[i];
        if (i != arr.size() - 1)
            std::cout << ", "; }
    std::cout << "]" << std::endl;

    const auto block = SharedExpQuantization<32, fp8::E5M2Type>::Quantize(arr); // test>
    std::cout << "Quantized array : " << block.as_string() << std::endl;
=======
    const auto block = MaximumFractionalQuantization<4, 32, fp8::E4M3Type>::Quantize(arr);
    std::cout << block.as_string() << std::endl;
>>>>>>> acf041785458ac68402d7adbd76e74b5c4cf0ea2
}