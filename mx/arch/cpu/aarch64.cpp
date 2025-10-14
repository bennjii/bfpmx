//
// Created by Benjamin White on 10/10/2025.
//

template<typename T>
struct CPUArithmetic
{
    static auto Add(const T& lhs, const T& rhs) -> T
    {
        return lhs;
    }

    static auto Sub(const T& lhs, const T& rhs) -> T
    {
        return lhs;
    }

    static auto Mul(const T& lhs, const T& rhs) -> T
    {
        return lhs;
    }

    static auto Div(const T& lhs, const T& rhs) -> T
    {
        return lhs;
    }
};