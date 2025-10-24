//
// Created by Benjamin White on 13/10/2025.
//

#include <iostream>

#include "definition/prelude.h"

int main() {
    auto arr = std::to_array<f64, 4>({
        1.2f, 3.4f, 5.6f, 7.8f,
    });

    const auto block = MaximumFractionalQuantization<4, 4, fp8::E4M3Type>::Quantize(arr);
    std::cout << block.as_string() << std::endl;
}