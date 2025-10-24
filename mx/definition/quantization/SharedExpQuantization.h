#pragma once

#include "definition/alias.h"
#include "definition/block_float/block/Block.h"
#include "definition/block_float/repr/FloatRepr.h"

template<
    std::size_t Size,
    IFloatRepr Float
>

/* ----------------------------
*  SharedExpQuantization
*  ----------------------------
*/                             

class SharedExpQuantization
{
public:
    using BlockFmt = Block<
        32,
        Size,
        Float,
        CPUArithmetic
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

        f64 sharedExp = SharedExponent(Float::ExponentBits(), largestValue);
        std::cout << "sharedExp : " << sharedExp << std::endl;
        
        std::array<std::array<u8, Float::SizeBytes()>, Size> blockScaledFloats;

        f64 scaleFactor = std::pow(2.0, sharedExp);
        std::cout << "Scale Factor : " << scaleFactor << std::endl;
        //TODO: scale factor has to be stored and retrieved (for unquantiz.)

        
        for (int i = 0; i < Size; i++)
        {
            f64 scaledValue = vec[i] / scaleFactor;

            auto byteRepr = Float::Marshal(scaledValue);
            blockScaledFloats[i] = byteRepr;
        }

        

        // TODO: convert u32 into u8[]
        std::array<u8, 32> dummyArr;
        return BlockFmt(blockScaledFloats, dummyArr);
    }

    std::array<f64, Size> UnQuantize(BlockFmt &block)
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
        static f64 SharedExponent(u16 exponentBits, f64 highestValueAbsolute)
        {
            // emax = (2^E - 2) - Bias
            auto bias = Float::BiasValue();
            const f64 emax = (1 << exponentBits) - 2 - bias;

            std::cout << "expbits = " << exponentBits << std::endl;
            std::cout << "bias = " << u64(bias) << std::endl;
            std::cout << "emax = " << emax << std::endl;
            
            // shared_exponent = floor(log2(maxVal))- emax
            return std::floor(std::log2(highestValueAbsolute)) - emax;
        }};