//
// Created by Benjamin White on 23/10/2025.
//
#pragma once

#include "definition/alias.h"
#include "definition/block_float/block/Block.h"
#include "definition/block_float/repr/FloatRepr.h"

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
        CPUArithmetic
    >;

    static BlockFmt Quantize(std::array<f64, Size> &vec)
    {
        f64 largestValue = 0;
        for (int i = 0; i < Size; i++)
        {
            if (vec[i] > largestValue)
            {
                largestValue = vec[i];
            }
        }

        f64 scaleFactor = ScaleFactor(Float::ElementBits(), largestValue);
        //u32 scaleFactorInt = lround(scaleFactor);
        std::cout << "ScaleFactor : " << scaleFactor << std::endl;

        std::array<std::array<u8, Float::SizeBytes()>, Size> blockScaledFloats;

        for (int i = 0; i < Size; i++)
        {
            f64 scaledValue = round(vec[i] / scaleFactor);
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

    std::array<f64, Size> UnQuantize(BlockFmt &block)
    {
        std::array<f64, Size> blockUnscaledFloats;
        for (int i = 0; i < Size; i++)
        {
            auto packedFloat = block.At(i);
            f64 fullPrecision = Float::Unmarshall(packedFloat);
            blockUnscaledFloats[i] = fullPrecision * block.Scalar();
        }

        return blockUnscaledFloats;
    }


private:
    static f64 ScaleFactor(u16 QuantizationSizeBits, f64 HighestValueAbsolute)
    {
        return Float::BiasValue() / HighestValueAbsolute;
    }
};
