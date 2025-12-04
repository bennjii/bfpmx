//
// Created by Benjamin White on 13/10/2025.
//

#include <iostream>

#include "definition/prelude.h"

template<
    u16 E,
    u16 M,
    u16 S
>
void print_bits(FloatRepr<E, M, S> repr, f64 value) {
    auto bits = repr.Marshal(value);
    const double decoded = repr.Unmarshal(bits);

    std::cout << "Value: " << value
              << " | Decoded: " << decoded
              << std::endl;

    // next representable in your custom FP
    const double next = repr.Next(value);
    if (next < value)
    {
        std::cout << "No further representable value." << std::endl;
        return;
    }

    std::cout << "Next representable: " << next << std::endl;
}

int main() {
    constexpr auto repr = fp16::E5M10;

    while (true) {
        std::cout << "Enter value: ";
        if (double val; std::cin >> val) {
            print_bits(repr, val);
        } else {
            break;
        }
    }
}
