//
// Created by Benjamin White on 10/10/2025.
//
#include "definition/alias.h"
#include <format>
#include <type_traits> // for std::remove_cvref_t

template<typename T>
struct CPUArithmetic
{
    static auto Add(const T& lhs, const T& rhs) -> T
    {
        using ElemType = f64;
        std::array<ElemType, T::dataCount()> result;

        using BlockType = std::remove_cvref_t<decltype(lhs)>;
        auto l = BlockType::QuantizationPolicyType::UnQuantize(lhs);
        auto r = BlockType::QuantizationPolicyType::UnQuantize(rhs);

        for (std::size_t i = 0; i < T::dataCount(); ++i)
            result[i] = l[i] + r[i];

        return T(result);
    }

    static auto Sub(const T& lhs, const T& rhs) -> T
    {
        using ElemType = f64;
        std::array<ElemType, T::dataCount()> result;

        using BlockType = std::remove_cvref_t<decltype(lhs)>;
        auto l = BlockType::QuantizationPolicyType::UnQuantize(lhs);
        auto r = BlockType::QuantizationPolicyType::UnQuantize(rhs);

        for (std::size_t i = 0; i < T::dataCount(); ++i)
            result[i] = l[i] - r[i];

        return T(result);
    }

    static auto Mul(const T& lhs, const T& rhs) -> T
    {
        using ElemType = f64;
        std::array<ElemType, T::dataCount()> result;

        using BlockType = std::remove_cvref_t<decltype(lhs)>;
        auto l = BlockType::QuantizationPolicyType::UnQuantize(lhs);
        auto r = BlockType::QuantizationPolicyType::UnQuantize(rhs);

        for (std::size_t i = 0; i < T::dataCount(); ++i)
            result[i] = l[i] * r[i];

        return T(result);
    }

    static auto Div(const T& lhs, const T& rhs) -> T
    {
        using ElemType = f64;
        std::array<ElemType, T::dataCount()> result;

        using BlockType = std::remove_cvref_t<decltype(lhs)>;
        auto l = BlockType::QuantizationPolicyType::UnQuantize(lhs);
        auto r = BlockType::QuantizationPolicyType::UnQuantize(rhs);

        for (std::size_t i = 0; i < T::dataCount(); ++i)
            result[i] = l[i] / r[i];

        return T(result);
    }
};