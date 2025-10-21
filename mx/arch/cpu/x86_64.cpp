//
// Created by Benjamin White on 10/10/2025.
//
#include "definition/alias.h"
#include <format>

template<typename T>
struct CPUArithmetic
{
    static auto Add(const T& lhs, const T& rhs) -> T
    {
        // NOTE(Gabri) just leave here the duplicate code, since once we use simd we have to do it anyway
        using ElemType = decltype(T::FloatType::Unmarshal(lhs.data()[0]) + T::FloatType::Unmarshal(rhs.data()[0]));
        std::array<ElemType, T::dataCount()> result;
        auto l = lhs.data(), r = lhs.data();
        for (std::size_t i = 0; i < T::dataCount(); ++i) {
            result[i] = T::FloatType::Unmarshal(l[i]) + T::FloatType::Unmarshal(r[i]);
        }
        return T(result);
    }

    static auto Sub(const T& lhs, const T& rhs) -> T
    {
        // NOTE(Gabri) just leave here the duplicate code, since once we use simd we have to do it anyway
        using ElemType = decltype(T::FloatType::Unmarshal(lhs.data()[0]) - T::FloatType::Unmarshal(rhs.data()[0]));
        std::array<ElemType, T::dataCount()> result;
        auto l = lhs.data(), r = lhs.data();
        for (std::size_t i = 0; i < T::dataCount(); ++i) {
            result[i] = T::FloatType::Unmarshal(l[i]) - T::FloatType::Unmarshal(r[i]);
        }
        return T(result);
    }

    static auto Mul(const T& lhs, const T& rhs) -> T
    {
        // NOTE(Gabri) just leave here the duplicate code, since once we use simd we have to do it anyway
        using ElemType = decltype(T::FloatType::Unmarshal(lhs.data()[0]) * T::FloatType::Unmarshal(rhs.data()[0]));
        std::array<ElemType, T::dataCount()> result;
        auto l = lhs.data(), r = lhs.data();
        for (std::size_t i = 0; i < T::dataCount(); ++i) {
            result[i] = T::FloatType::Unmarshal(l[i]) * T::FloatType::Unmarshal(r[i]);
        }
        return T(result);
    }

    static auto Div(const T& lhs, const T& rhs) -> T
    {
        // NOTE(Gabri) just leave here the duplicate code, since once we use simd we have to do it anyway
        using ElemType = decltype(T::FloatType::Unmarshal(lhs.data()[0]) / T::FloatType::Unmarshal(rhs.data()[0]));
        std::array<ElemType, T::dataCount()> result;
        auto l = lhs.data(), r = lhs.data();
        for (std::size_t i = 0; i < T::dataCount(); ++i) {
            result[i] = T::FloatType::Unmarshal(l[i]) / T::FloatType::Unmarshal(r[i]);
        }
        return T(result);
    }
};
