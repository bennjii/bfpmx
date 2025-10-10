//
// Created by Benjamin White on 10/10/2025.
//
#pragma once

#ifndef BFPMX_FLOATREPR_H
#include "definition/alias.h"
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
template<
    u16 Significand,
    u16 Exponent,
    u16 Sign,
    u16 Bias
>
class FloatRepr
{
public:
    [[nodiscard]] static constexpr u8 SignificandBits() {
        return Significand;
    }

    [[nodiscard]] static constexpr u8 ExponentBits() {
        return Exponent;
    }

    [[nodiscard]] static constexpr u8 SignBits() {
        return Sign;
    }

    [[nodiscard]] static constexpr u8 ElementBits()
    {
        return
            + SignificandBits()
            + ExponentBits()
            + SignBits();
    }

    [[nodiscard]] static constexpr u32 Size()
    {
        return ElementBits();
    }
};

namespace fp
{
    using E4M3Type = FloatRepr<3, 4, 1, 7>;
    constexpr auto E4M3 = FloatRepr<3, 4, 1, 7>();
}

#endif //BFPMX_FLOATREPR_H