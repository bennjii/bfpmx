//
// Created by Benjamin White on 10/10/2025.
//

#ifndef BFPMX_ARITHMETIC_H
#define BFPMX_ARITHMETIC_H


template<typename T>
class IArithmetic
{
public:
    virtual ~IArithmetic() = default;  // Always make base destructors virtual

    // Other type must need to conform to some given interface,
    // so it supports float32, float64, ...
    virtual double add(T other);
    virtual double subtract(T other);
};


#endif //BFPMX_ARITHMETIC_H