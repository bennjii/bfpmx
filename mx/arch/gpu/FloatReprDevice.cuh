// device-friendly helper: pack form
#pragma once

struct PackedFormDevice {
    unsigned long long sign;
    unsigned long long exp;
    unsigned long long frac;
};

constexpr u16 F64_SIGNIFICAND_ = 52;
constexpr u16 F64_BIAS_ = 1023;

template <uint8_t Exponent, uint8_t Significand, uint8_t Sign>
struct FloatReprDevice {
    // constants copied from your CPU version
    static __device__ __forceinline__ unsigned char SignificandBits() { return Significand; }
    static __device__ __forceinline__ unsigned char ExponentBits()    { return Exponent; }
    static __device__ __forceinline__ unsigned char SignBits()        { return Sign; }

    static __device__ __forceinline__ unsigned char BiasValue() {
        return (1 << (Exponent - 1)) - 1;
    }

    static __device__ __forceinline__ uint8_t ElementBits() {
        return SignificandBits() + ExponentBits() + SignBits();
    }

    static __device__ __forceinline__ unsigned int SizeBytes() {
        return ElementBits() / 8;
    }

    static __device__ __forceinline__ double UnpackDevice(PackedFormDevice v) {
        unsigned long long sign = v.sign;
        unsigned long long exp  = v.exp;
        unsigned long long frac = v.frac;

        unsigned long long f64Sign = sign << 63;
        unsigned long long f64Exp;
        unsigned long long f64Frac;

        if (exp == 0) {
            if (frac == 0) {
                f64Exp = 0;
                f64Frac = 0;
            } else {
                f64Exp = 0;
                f64Frac = frac << (F64_SIGNIFICAND_ - SignificandBits());
            }
        } else if (exp == ((1u << ExponentBits()) - 1)) {
            f64Exp  = 0x7FFull << F64_SIGNIFICAND_;
            f64Frac = frac ? 1ull : 0ull;
        } else {
            const long long e = (long long)exp - BiasValue() + F64_BIAS_;
            f64Exp  = ((unsigned long long)e) << F64_SIGNIFICAND_;
            f64Frac = frac << (F64_SIGNIFICAND_ - SignificandBits());
        }

        unsigned long long f64Bits = f64Sign | f64Exp | f64Frac;

        union {
            unsigned long long u;
            double d;
        } caster;
        caster.u = f64Bits;
        return caster.d;
    }

    static __device__ __forceinline__ double UnmarshalDevice(const unsigned char* bytes) {
        unsigned long long bits = 0;
        #pragma unroll
        for (unsigned int i = 0; i < SizeBytes(); ++i) {
            bits |= (unsigned long long)bytes[i] << (8 * i);
        }

        const unsigned long long fracMask = (1ull << SignificandBits()) - 1;
        unsigned long long frac = bits & fracMask;

        const unsigned long long expMask = (1ull << ExponentBits()) - 1;
        unsigned long long exp = (bits >> SignificandBits()) & expMask;

        unsigned long long sign = (bits >> (SignificandBits() + ExponentBits())) & 1ull;

        PackedFormDevice pf{sign, exp, frac};
        return UnpackDevice(pf);
    }
};