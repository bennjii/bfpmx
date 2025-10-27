#ifndef BFPMX_SEQ_H
#define BFPMX_SEQ_H

#include "definition/alias.h"
#include "definition/block_float/block/Block.h"
#include "definition/block_float/repr/FloatRepr.h"

#include <iostream>

template<
    std::size_t ScalarBytes,
    std::size_t Size,
    IFloatRepr Float
>
class SharedExponentQuantization
{
public:
    using BlockFmt = Block<
        ScalarBytes,
        Size,
        Float,
        CPUArithmetic,
        SharedExponentQuantization
    >;
    using PackedFloat = std::array<u8, Float::SizeBytes()>;

    static BlockFmt Quantize(std::array<f64, Size> &vec)
    {
        u64 largestBiasedExponent = 0;
        for (int i = 0; i < Size; i++)
        {
            u64 bits;
            std::memcpy(&bits, &vec[i], sizeof(bits));
            const u64 exponent = (bits >> 52) & 0x7FF;

            if (exponent > largestBiasedExponent) {
                largestBiasedExponent = exponent;
            }
        }

        const u64 largestUnbiasedExponent = largestBiasedExponent - 1023;
        const u64 exponent = NormalizedExponent(largestUnbiasedExponent);
        f64 scaleFactor = std::pow(2.0, exponent);

        std::array<PackedFloat, Size> blockScaledFloats;
        for (int i = 0; i < Size; i++)
        {
            f64 scaledValue = vec[i] / scaleFactor;
            PackedFloat packed = Float::Marshal(scaledValue);
            blockScaledFloats[i] = packed;
        }

        const u32 scaleFactorInt = lround(exponent);

        std::array<u8, ScalarBytes> packedScalar;
        for (int i = 0; i < ScalarBytes; i++)
        {
            packedScalar[i] = static_cast<u8>(scaleFactorInt >> (i * 8));
        }

        return BlockFmt(blockScaledFloats, packedScalar);
    }

    static std::array<f64, Size> UnQuantize(const BlockFmt &block)
    {
        std::array<f64, Size> blockUnscaledFloats;
        for (int i = 0; i < Size; i++)
        {
            auto packedFloat = block.At(i);
            f64 fullPrecision = Float::Unmarshal(packedFloat);
            blockUnscaledFloats[i] = fullPrecision * block.Scalar();
        }

        return blockUnscaledFloats;
    }


private:
    static constexpr u64 MaximumScalarExponentValue()
    {
        return (static_cast<u64>(1) << (ScalarBytes * 8)) - 1;
    }

    static u64 NormalizedExponent(const u64 highestExponent)
    {
        // Exponent cannot exceed the representable maximum.
        return std::min(MaximumScalarExponentValue(), highestExponent);
    }
};

#endif // BFPMX_SEQ_H