//
// Created by Benjamin White on 10/10/2025.
//
#pragma once

#ifndef BFPMX_FLOATREPR_H

#include "definition/alias.h"

#include <concepts>
#include <array>
#include <string>
#include <sstream>

#define BFPMX_FLOATREPR_H

constexpr u16 F64_BITS_EXPONENT = 11;
constexpr u16 F64_BITS_SIGNIFICAND = 52;
constexpr u16 F64_BIAS = 1023;

template<typename T>
concept IFloatRepr = requires(f64 v, std::array<u8, T::SizeBytes()> a) {
    { T::SizeBytes() } -> std::convertible_to<u32>;
    { T::Marshal(v) } -> std::same_as<std::array<u8, T::SizeBytes()>>;
    { T::Unmarshal(a) } -> std::convertible_to<f64>;

    { T::ElementBits() } -> std::convertible_to<u32>;
    { T::SignificandBits() } -> std::convertible_to<u32>;
    { T::ExponentBits() } -> std::convertible_to<u32>;
    { T::SignBits() } -> std::convertible_to<u32>;

    // TODO: Add the following functions:
    //  - Max() -> f64
    //  - Min() -> f64
    //  - Bias() -> f64
    //  - ...
    //  We need to understand what the min/max values
    //  we can represent are, in order to perform quant*.
};

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
    u16 Exponent,
    u16 Significand,
    u16 Sign
>
class FloatRepr
{
public:
    using PackedForm = std::tuple<u64 /*sign*/, u64 /*exponent*/, u64 /*fraction*/>;

    [[nodiscard]] static constexpr std::string Nomenclature()
    {
        std::ostringstream oss;
        oss << "FP" << +ElementBits()    // the + ensures u8 prints as a number
            << " "
            << "E" << +ExponentBits()
            << "M" << +SignificandBits();
        return oss.str();
    }

    static constexpr f64 Next(const f64 value) {
        if (value == 0.0) {
            // smallest positive subnormal
            return std::pow(2.0, 1-BiasValue()) / (1 << SignificandBits());
        }

        auto [sign, exp, frac] = Pack(value); // you already have this info in Marshal
        if (sign == 0) { // positive
            if (frac < (1u << SignificandBits()) - 1) {
                ++frac;
            } else {
                frac = 0;
                ++exp;
            }
        } else { // negative
            // just return previous magnitude
            if (frac > 0) --frac;
            else { frac = (1<<SignificandBits())-1; --exp; }
        }

        return Unpack({sign, exp, frac});
    }

    [[nodiscard]] static constexpr u8 SignificandBits() {
        return Significand;
    }

    [[nodiscard]] static constexpr u8 ExponentBits() {
        return Exponent;
    }

    [[nodiscard]] static constexpr u8 SignBits() {
        return Sign;
    }

    [[nodiscard]] static constexpr u8 BiasValue() {
        return (1 << (Exponent - 1)) - 1;
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

    [[nodiscard]] static constexpr u32 SizeBytes()
    {
        // TODO: this way we are not able to store in a compact way 6 or 4 bits floats (e.g. E2M3, E3M2, E2M1)in a Block
        //       we still use 8 bits per significan instead of 6 (or 4).
        return Size() / 8;
    }

    [[nodiscard]] static constexpr PackedForm Pack(const f64 value)
    {
        constexpr u16 srcExpBits = 11;
        constexpr u16 srcSigBits = 52;

        // reinterpret the double bits
        const u64 bits = std::bit_cast<u64>(value);

        const u64 sign = (bits >> 63) & 0x1;
        const i64 exp  = ((bits >> srcSigBits) & ((1ull << srcExpBits) - 1));
        const u64 frac = bits & ((1ull << srcSigBits) - 1);

        // handle special cases
        u32 newSign = static_cast<u32>(sign);
        u32 newExp;
        u64 newFrac;

        if (exp == 0x7FF) { // NaN or Inf
            newExp  = (1u << ExponentBits()) - 1;
            newFrac = frac ? 1 : 0; // canonicalize NaN
        }
        else if (exp == 0) { // subnormal or zero
            newExp  = 0;
            newFrac = 0;
        }
        else
        {
            constexpr u16 srcBias = 1023;
            // normalize exponent to target bias
            if (const i64 e = exp - srcBias + BiasValue(); e <= 0) {  // underflow
                newExp  = 0;
                newFrac = 0;
            } else if (e >= ((1 << ExponentBits()) - 1)) { // overflow
                newExp  = (1u << ExponentBits()) - 1;
                newFrac = 0;
            } else {
                newExp  = static_cast<u32>(e);
                // round mantissa down to target width
                newFrac = frac >> (srcSigBits - SignificandBits());
            }
        }

        return {newSign, newExp, newFrac};
    }

    [[nodiscard]] static constexpr f64 Unpack(PackedForm value)
    {
        const auto [sign, exp, frac] = value;

        const u64 f64Sign = sign << 63;
        u64 f64Exp;
        u64 f64Frac;

        if (exp == 0) {
            if (frac == 0) {
                // zero
                f64Exp = 0;
                f64Frac = 0;
            } else {
                // subnormal (scale fraction)
                f64Exp = 0;
                f64Frac = frac << (F64_BITS_SIGNIFICAND - SignificandBits());
            }
        } else if (exp == ((1u << ExponentBits()) - 1)) {
            // inf or NaN
            f64Exp = 0x7FFull << F64_BITS_SIGNIFICAND;
            f64Frac = frac ? 1ull : 0;
        } else {
            // normal
            const i64 e = static_cast<i64>(exp) - BiasValue() + F64_BIAS;
            f64Exp = static_cast<u64>(e) << F64_BITS_SIGNIFICAND;
            f64Frac = frac << (F64_BITS_SIGNIFICAND - SignificandBits());
        }

        const u64 f64Bits = f64Sign | f64Exp | f64Frac;
        return std::bit_cast<f64>(f64Bits);
    }

    [[nodiscard]] static constexpr f64 Unmarshal(std::array<u8, SizeBytes()> v)
    {
        u64 bits = 0;
        for (u32 i = 0; i < SizeBytes(); ++i)
            bits |= static_cast<u64>(v[i]) << (8 * i);

        const u64 fracMask = (1ull << SignificandBits()) - 1;
        const u64 frac = bits & fracMask;

        const u64 expMask = (1ull << ExponentBits()) - 1;
        const u64 exp  = (bits >> SignificandBits()) & expMask;

        const u64 sign = (bits >> (SignificandBits() + ExponentBits())) & 1;

        return Unpack({sign, exp, frac});
    }

    [[nodiscard]] static constexpr std::array<u8, SizeBytes()> Marshal(const f64 value)
    {
        auto [sign, exp, frac] = Pack(value);

        // pack into bits
        const u64 encoded =
            (sign  << (ExponentBits() + SignificandBits())) |
            (exp   << SignificandBits()) |
            (frac & ((1ull << SignificandBits()) - 1));

        // store to bytes
        std::array<u8, SizeBytes()> out{};
        for (u32 i = 0; i < SizeBytes(); ++i)
            out[i] = static_cast<u8>((encoded >> (8 * i)) & 0xFF);
        return out;
    }
};

namespace fp8
{
    using E4M3Type = FloatRepr<4, 3, 1>;
    constexpr auto E4M3 = E4M3Type();

    using E5M2Type = FloatRepr<5, 2, 1>;
    constexpr auto E5M2 = E5M2Type();
}

namespace fp6
{
    using E2M3Type = FloatRepr<2, 3, 1>;
    constexpr auto E2M3 = E2M3Type();

    using E3M2Type = FloatRepr<3, 2, 1>;
    constexpr auto E3M2 = E3M2Type();
}

namespace fp4
{
    using E2M1Type = FloatRepr<2, 1, 1>;
    constexpr auto E2M1 = E2M1Type();
}

#endif //BFPMX_FLOATREPR_H
