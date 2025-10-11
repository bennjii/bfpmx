//
// Created by Benjamin White on 10/10/2025.
//

template<typename T>
struct CPUArithmetic
{
    static T Add(const T& l, const T& r)
    {
        return l;
    }

    static T Sub(const T& l, const T& r)
    {
        return l;
    }

    static T Mul(const T& l, const T& r)
    {
        return l * r;
    }

    static T Div(const T& l, const T& r)
    {
        return l / r;
    }
};