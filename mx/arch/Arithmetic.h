//
// Created by Benjamin White on 10/10/2025.
//

#ifndef BFPMX_ARITHMETIC_H
#define BFPMX_ARITHMETIC_H

template<typename Impl, typename T>
concept ArithmeticImpl = requires(const T& a, const T& b) {
    Impl::Add(a, b);            // expression must be valid
};

template<typename T, typename Impl>
struct ArithmeticEnabled {
    friend T operator+(const T& lhs, const T& rhs)
        requires ArithmeticImpl<Impl, T>
    {
        return Impl::Add(lhs, rhs);
    }

    friend T operator-(const T& lhs, const T& rhs)
        requires ArithmeticImpl<Impl, T>
    {
        return Impl::Sub(lhs, rhs);
    }

    friend T operator*(const T& lhs, const T& rhs)
        requires ArithmeticImpl<Impl, T>
    {
        return Impl::Mul(lhs, rhs);
    }

    friend T operator/(const T& lhs, const T& rhs)
        requires ArithmeticImpl<Impl, T>
    {
        return Impl::Div(lhs, rhs);
    }
};

#endif //BFPMX_ARITHMETIC_H