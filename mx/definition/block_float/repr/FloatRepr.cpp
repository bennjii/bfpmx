//
// Created by Benjamin White on 10/10/2025.
//

#include "FloatRepr.h"

FloatRepr::FloatRepr(const u8 significand, const u8 exponent, const u8 sign_bit, const u8 bias)
{
    this->bits_significand = significand;
    this->bits_exponent = exponent;
    this->bits_sign = sign_bit;
    this->exponent_bias = bias;
}

FloatRepr FloatRepr::EMB(const u8 exponent, const u8 mantissa, const u8 bias)
{
    constexpr u8 DEFAULT_SIGN_BITS = 1;
    return FloatRepr(mantissa, exponent, DEFAULT_SIGN_BITS, bias);
}

FloatRepr FloatRepr::E4M3()
{
    // Page 11 of OCP's MX Format Spec
    return EMB(4, 3, 7);
}

FloatRepr FloatRepr::E5M2()
{
    return EMB(5, 2, 15);
}
