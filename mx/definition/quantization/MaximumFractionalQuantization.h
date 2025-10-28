//
// Created by Benjamin White on 23/10/2025.
//

#ifndef BFPMX_MFQ_H
#define BFPMX_MFQ_H

#include "definition/alias.h"
#include "definition/block_float/block/Block.h"
#include "definition/block_float/repr/FloatRepr.h"
#include "arch/cpu/CPUArithmetic.h"
#include <iostream>

template<
    std::size_t ScalarBytes,
    BlockDimsType BlockShape,
    IFloatRepr Float
>
class MaximumFractionalQuantization
{
public:
    using BlockFmt = Block<
        ScalarBytes,
        BlockShape,
        Float,
        CPUArithmetic,
        MaximumFractionalQuantization
    >;
    static BlockFmt Quantize(std::array<f64, BlockShape::total_size()> &vec)
    {
        f64 largestValue = 0;
        for (int i = 0; i < BlockShape::total_size(); i++)
        {
            if (const f64 absValue = fabs(vec[i]); absValue > largestValue)
            {
                largestValue = absValue;
            }
        }

        const u32 scaleFactorCandidate = lround(largestValue);
        const u32 scaleFactorInt = 31 - __builtin_clz(scaleFactorCandidate);
        const u32 scaleFactor = 1 << scaleFactorInt;

        std::array<std::array<u8, Float::SizeBytes()>, BlockShape::total_size()> blockScaledFloats;

        for (int i = 0; i < BlockShape::total_size(); i++)
        {
            f64 scaledValue = vec[i] / scaleFactor;
            blockScaledFloats[i] = Float::Marshal(scaledValue);
        }

        std::array<u8, ScalarBytes> packedScalar;
        for (int i = 0; i < ScalarBytes; i++)
        {
            packedScalar[i] = scaleFactorInt >> (i * 8);
        }

        return BlockFmt(blockScaledFloats, packedScalar);
    }

    static std::array<f64, BlockShape::total_size()> UnQuantize(const BlockFmt &block)
    {
        std::array<f64, BlockShape::total_size()> blockUnscaledFloats;
        for (int i = 0; i < BlockShape::total_size(); i++)
        {
            auto packedFloat = block.At(i);
            f64 fullPrecision = Float::Unmarshal(packedFloat);
            blockUnscaledFloats[i] = fullPrecision * block.Scalar();
        }

        return blockUnscaledFloats;
    }


private:
    static f64 ScaleFactor(f64 HighestValueAbsolute)
    {
        return HighestValueAbsolute / Float::BiasValue();
    }
};

#endif //BFPMX_MFQ_H