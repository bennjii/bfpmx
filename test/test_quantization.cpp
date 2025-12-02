//
// Created by Benjamin White on 29/10/2025.
//

#include <cassert>

#define PROFILE 1

#include "definition/prelude.h"
#include "helper/test.h"
#include "profiler/profiler.h"

// Constants to be used for the testing regime
using BlockScalar = u32; // u32 (4 bytes)
using BlockSize = BlockDims<32>;
using BlockFloat = fp8::E4M3Type;

template<
    template <typename T> typename ArithmeticPolicy,
    template <std::size_t, BlockDimsType, IFloatRepr> typename QuantizationPolicy
>
using BlockFmt = Block<BlockScalar, BlockSize, BlockFloat, ArithmeticPolicy, QuantizationPolicy>;

template<
    template <typename T> typename A,
    template <std::size_t, BlockDimsType, IFloatRepr> typename Q
> BlockFmt<A, Q> New(const std::array<f64, BlockSize::TotalSize()> &full_precision) {
    return BlockFmt<A, Q>::Quantize(full_precision);
}

template<
    template <typename T> typename A,
    template <std::size_t, BlockDimsType, IFloatRepr> typename Q
>
void Test(const std::array<f64, BlockSize::TotalSize()> &full_precision) {
    using Quantizer = Q<sizeof(BlockScalar), BlockSize, BlockFloat>;

    BlockFmt<A, Q> block = New<A, Q>(full_precision);

    for (int i = 0; i < BlockSize::TotalSize(); ++i) {
        f64 recoveredValue = block.RealizeAtUnsafe(i);
        f64 originalValue = full_precision.at(i);

        const bool isEquivalent = FuzzyEqual<BlockFloat>(recoveredValue, originalValue, 5);
        if (!isEquivalent) {
            const std::string equalityString = std::format("Expected {:.9f}, but got {:.9f}", originalValue, recoveredValue);
            std::cerr << "Original value and recovered value are not equivalent: " << equalityString << std::endl;
            std::cerr << "For index " << +i << " using quantization policy " << Quantizer::Identity() << std::endl;
            std::cerr << "The block had the scalar value: " << block.ScalarValue() << std::endl;
            std::cerr << "The float at the location was: " << block.RealizeAtUnsafe(i) << std::endl;

            assert(false);
        }
    }
}

// Write discrete testing functions for each form of quantization.
// This can then be performed over each form of arithmetic available
// to the local machine.
template<
    template <typename T> typename A
>
void TestAllQuantization(const std::array<f64, BlockSize::TotalSize()> &full_precision)
{
    {
        profiler::block("L2 Norm Quantization");
        Test<A, L2NormQuantization>(full_precision);
    }

    {
        profiler::block("Shared Exponent Quantization");
        Test<A, SharedExponentQuantization>(full_precision);
    }

    {
        profiler::block("Maximal Fractional Quantization");
        Test<A, MaximumFractionalQuantization>(full_precision);
    }
}

// Define elsewhere.
#define CPU_COMPATIBLE

void TestAllArithmetic(const std::array<f64, BlockSize::TotalSize()> &full_precision) {
#ifdef CPU_COMPATIBLE
    TestAllQuantization<CPUArithmetic>(full_precision);
    TestAllQuantization<CPUArithmeticWithoutMarshalling>(full_precision);
#endif

#ifdef HAS_CUDA
    TestAllQuantization<GPUArithmetic>(full_precision);
#endif
}

int main() {
    profiler::begin();

    constexpr u64 iterations = 1000;

    constexpr f64 min = -10.0;
    constexpr f64 max = +10.0;

    const std::array array =
      fill_random_arrays<f64, BlockSize::TotalSize()>(min, max);

    for (int i = 0; i < iterations; i++) {
        TestAllArithmetic(array);
    }

    profiler::end_and_print();
}