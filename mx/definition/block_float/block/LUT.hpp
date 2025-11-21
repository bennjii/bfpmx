#pragma once 

constexpr float fp8_e4m3_to_f32(u8 v) {
    // Inline your Unmarshal + Unpack logic here,
    // but using only constexpr-friendly operations.
    const u64 bits = v;

    const u64 fracMask = (1ull << SignificandBits()) - 1;
    const u64 frac = bits & fracMask;

    const u64 expMask = (1ull << ExponentBits()) - 1;
    const u64 exp = (bits >> SignificandBits()) & expMask;

    const u64 sign = (bits >> (SignificandBits() + ExponentBits())) & 1;

    return static_cast<float>(Unpack({sign, exp, frac}));
}