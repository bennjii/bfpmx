//
// Created by Benjamin White on 23/10/2025.
//
#pragma once

#include "definition/alias.h"
#include "definition/block_float/block/Block.h"
#include "definition/block_float/repr/FloatRepr.h"
#include "arch/cpu/arithmetic.hpp"
template<
    std::size_t ScalarBytes,
    std::size_t Size,
    IFloatRepr Float
>
class MaximumFractionalQuantization
{
public:
    using BlockFmt = Block<
        ScalarBytes,
        Size,
        Float,
        CPUArithmetic,
        MaximumFractionalQuantization
    >;
    static BlockFmt Quantize(std::array<f64, Size> &vec)
    {
        f64 largestValue = 0;
        for (int i = 0; i < Size; i++)
        {
            if (abs(vec[i]) > abs(largestValue))
            {
                largestValue = vec[i];
            }
        }

        f64 scaleFactor = ScaleFactor(Float::ElementBits(), largestValue);
        u32 scaleFactorInt = lround(scaleFactor);

        std::array<std::array<u8, Float::SizeBytes()>, Size> blockScaledFloats;

        for (int i = 0; i < Size; i++)
        {
            f64 scaledValue = vec[i] / scaleFactor;
            blockScaledFloats[i] = Float::Marshal(scaledValue);
            std::cerr << "[Quant] scaledValue: " << scaledValue << " "
            << vec[i] << " " << scaleFactor << std::endl;
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
            auto packedFloat = block.data()[i];
            f64 fullPrecision = Float::Unmarshal(packedFloat);
            blockUnscaledFloats[i] = fullPrecision * block.Scalar();
        }

        return blockUnscaledFloats;
    }


private:
    static f64 ScaleFactor(u16 QuantizationSizeBits, f64 HighestValueAbsolute)
    {
        return HighestValueAbsolute / Float::BiasValue();
    }
};
