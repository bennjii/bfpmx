//
// Created by Benjamin White on 10/10/2025.
//
#pragma once

#ifndef BFPMX_FLOATREPR_H
#include "../../alias.cpp"
#define BFPMX_FLOATREPR_H

/**
 *
 * Mantissa is also known as the significand (2nd argument).
 *
 * ### Sign Calculation
 * The sign of the value is given by the function:
 *
 *     sign = (-1) ^ S
 *
 * Where, v = 1 when the sign is positive, such that
 * S is 0, or v = -1 when the sign represents a negative value.
 * Wherein, S is 1.
 *
 * ### Exponent
 *
 * When the value of the exponent is 0 (E == 0), we are representing
 * a subnormal number. Otherwise, we are representing a normal
 * number.
 *
 * If E == 0, then our scaling is represented by.
 *
 *     exponent = 2 ^ (1 - B)
 *
 * However, if E > 0, then our scaling is represented by:
 *
 *     exponent = 2 ^ (E - B)
 *
 * Where, E is the value of the exponent, and B is the bias
 * value. This is customised for each representation.
 *
 * Ultimately, this controls how large or small the fractional values
 * can be.
 *
 * ### Mantissa
 *
 * Lastly, we have the Mantissa, or the Significand. This value
 * will provide the precision for our fraction. So, the more significand
 * bits we have, the more accurate our representation of numbers
 * between powers of two will be.
 *
 * The mantissa calculation is given by:
 *
 *     mantissa = (n + 2 ^ (-M_b)) * M
 *
 * Where, `n` is 1 when our exponent is greater than 0, otherwise
 * it is 0. The value M_b is the number of bits in our mantissa.
 *
 * Lastly, M is the mantissa itself. This is an N-bit integer value,
 * as specified in 5.3 of the OCP Microscaling Format Specification.
 *
 */
class FloatRepr
{
public:
    explicit FloatRepr(u8 significand, u8 exponent, u8 sign_bit, u8 bias);

    [[nodiscard]] u8 SignificandBits() const {
        return this->bits_significand;
    }

    [[nodiscard]] u8 ExponentBits() const {
        return this->bits_exponent;
    }

    [[nodiscard]] u8 SignBits() const {
        return this->bits_sign;
    }

    [[nodiscard]] u8 ElementBits() const
    {
        return
            + this->bits_significand
            + this->bits_exponent
            + this->bits_sign;
    }

    [[nodiscard]] u32 Size() const
    {
        return this->ElementBits();
    }

    /**
     *  Exponent-Mantissa (Bias) Form
     *
     * @param exponent Number of exponential bits
     * @param mantissa Number of significand/mantissa bits
     * @param bias The bias value
     *
     * @return A block float construction
     */
    [[nodiscard]] static FloatRepr EMB(u8 exponent, u8 mantissa, u8 bias);

    static FloatRepr E4M3();
    static FloatRepr E5M2();

private:
    // Represents the sizes of the
    // elements within the floating point block.
    u8 bits_significand;
    u8 bits_exponent;
    u8 bits_sign;

    // The bias value for the exponent
    u8 exponent_bias;
};


#endif //BFPMX_BLOCKFLOAT_H