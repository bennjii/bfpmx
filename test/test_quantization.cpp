//
// Created by Benjamin White on 29/10/2025.
//

#include <cassert>

#define PROFILE 1

#include "definition/prelude.h"
#include "profiler/profiler.h"

// Constants to be used for the testing regime
constexpr u32 BlockScalar = 4; // u32 (4 bytes)
using BlockSize = BlockDims<32>;
using BlockFloat = fp8::E4M3Type;

template<
    template <typename T> typename ArithmeticPolicy,
    template <std::size_t, BlockDimsType, IFloatRepr, template<typename> typename ArithmeticPolicy_> typename QuantizationPolicy
>
using BlockFmt = Block<BlockScalar, BlockSize, BlockFloat, ArithmeticPolicy, QuantizationPolicy>;

template<
    template <typename T> typename A,
    template <std::size_t, BlockDimsType, IFloatRepr, template<typename> typename A_> typename Q
> BlockFmt<A, Q> New(const std::array<f64, BlockSize::TotalSize()> &full_precision) {
    return BlockFmt<A, Q>::Quantize(full_precision);
}

template<
    template <typename T> typename A,
    template <std::size_t, BlockDimsType, IFloatRepr, template<typename> typename A_> typename Q
>
void Test(const std::array<f64, BlockSize::TotalSize()> &full_precision) {
    using Quantizer = Q<BlockScalar, BlockSize, BlockFloat, A>;

    BlockFmt<A, Q> block = New<A, Q>(full_precision);

    for (int i = 0; i < BlockSize::TotalSize(); ++i) {
        f64 recoveredValue = block.RealizeAtUnsafe(i);
        f64 originalValue = full_precision.at(i);

        const bool isEquivalent = FuzzyEqual<BlockFloat>(recoveredValue, originalValue);
        if (!isEquivalent) {
            const std::string equalityString = std::format("Expected {:.9f}, but got {:.9f}", originalValue, recoveredValue);
            std::cerr << "Original value and recovered value are not equivalent: " << equalityString << std::endl;
            std::cerr << "For index " << +i << " using quantization policy " << Quantizer::Identity() << std::endl;

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
#endif

#ifdef HAS_CUDA
    TestAllQuantization<GPUArithmetic>(full_precision);
#endif
}

int main() {
    profiler::begin();

    constexpr std::array<f64, BlockSize::TotalSize()> EXAMPLE_ARRAY =
        std::to_array<f64, BlockSize::TotalSize()>({
            1.2f, 3.4f, 5.6f, 2.1f, 1.3f, -6.5f
        });

    for (int i = 0; i < 100000; i++) {
        TestAllArithmetic(EXAMPLE_ARRAY);
    }

    profiler::end_and_print();
}