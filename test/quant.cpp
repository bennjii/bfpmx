//
// Created by Benjamin White on 13/10/2025.
//

#include <iostream>

#include "definition/prelude.h"

int main() {
    std::array<f64, 32> arr = std::to_array<f64, 32>({
        1.2f, 3.4f, 5.6f
    });

    const auto block = MaximumFractionalQuantization<4, 32, fp8::E4M3Type>::Quantize(arr);
    std::cout << block.as_string() << std::endl;
}