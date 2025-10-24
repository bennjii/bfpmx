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

    static BlockFmt Quantize(std::array<f64, Size> &vec)
    {
        f64 largestValue = 0;
        for (int i = 0; i < Size; i++)
        {
            if (std::abs(vec[i]) > largestValue)
            {
                largestValue = std::abs(vec[i]);
            }
        }

        const f64 sharedExp = SharedExponent(largestValue);
        std::array<std::array<u8, Float::SizeBytes()>, Size> blockScaledFloats;

        f64 scaleFactor = std::pow(2.0, sharedExp);
        const u32 scaleFactorInt = lround(log2(scaleFactor));

        for (int i = 0; i < Size; i++)
        {
            f64 scaledValue = vec[i] / scaleFactor;

            auto byteRepr = Float::Marshal(scaledValue);
            blockScaledFloats[i] = byteRepr;
        }

        std::array<u8, ScalarBytes> packedScalar;
        for (int i = 0; i < ScalarBytes; i++)
        {
            packedScalar[i] = scaleFactorInt >> (i * 8);
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
    // TODO: Does not find the largest shared exponent optimally.
    static f64 SharedExponent(const f64 highestValueAbsolute)
    {
        // emax = (2^E - 2) - Bias
        auto bias = Float::BiasValue();
        const f64 emax = (1 << Float::ExponentBits()) - 2 - bias;

        std::cout << "expbits = " << Float::ExponentBits() << std::endl;
        std::cout << "bias = " << u64(bias) << std::endl;
        std::cout << "emax = " << emax << std::endl;

        // shared_exponent = floor(log2(maxVal))- emax
        return std::floor(std::log2(highestValueAbsolute)) - emax;
    }
};

#endif // BFPMX_SEQ_H