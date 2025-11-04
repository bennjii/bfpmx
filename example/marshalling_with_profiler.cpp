#include <format>
#include <iostream>
#include <ostream>

#define PROFILE 1
#include "../profiler/profiler.h"
#include "definition/prelude.h"

template<
    u16 E,
    u16 M,
    u16 S
>
void test_round_trip(FloatRepr<E, M, S> repr, f64 value, f64 range) {
    profiler::function();
    auto bytes = repr.Marshal(value);
    auto decoded = repr.Unmarshal(bytes);

    std::cout << "round_trip: " << repr.Nomenclature() << " Testing " << value << " to within " << range <<  " ... " << std::flush;

    if (std::abs(decoded - value) > range)
    {
        std::cout << "FAIL." << std::endl;

        std::cout << "\tOffBy=\t\t" << std::format("{:.3f}", std::abs(decoded - value)) << std::endl;
        std::cout << "\tExpectedWithin=\t" << std::format("{:.3f}", range) << std::endl;
        std::cout << "\tInput=\t\t" << std::format("{:.3f}", value) << std::endl;
        std::cout << "\tOutput=\t\t" << std::format("{:.3f}", decoded) << std::endl;

        // assert(decoded == value);
    } else
    {
        std::cout << "OK. " << "(Dec=" << decoded << ")" << std::endl;
    }
}

int main()
{
    profiler::begin();
    {
        profiler::block("first_three");
        test_round_trip(fp8::E5M2, 15, 1);
        test_round_trip(fp8::E4M3, 15, 0.1);
        test_round_trip(fp6::E2M3, 15, 1);
    }
    {
        profiler::block_bandwidth("remaining_two", 42000); // 42000 is just a random number
        test_round_trip(fp6::E3M2, 15, 0.1);
        test_round_trip(fp4::E2M1, -1, 1);
    }
    profiler::end_and_print();
    return 0;
}
