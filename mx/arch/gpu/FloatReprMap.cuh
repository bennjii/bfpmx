#pragma once

template<unsigned short E, unsigned short M, unsigned short S>
class FloatRepr;

// Device float repr
#include "arch/gpu/FloatReprDevice.cuh"

template <typename CPUFloatRepr>
struct DeviceFloatReprOf; // primary template

// Specialization that maps CPU FloatRepr<E,M,S> → GPU FloatReprDevice<E,M,S>
template <unsigned short E, unsigned short M, unsigned short S>
struct DeviceFloatReprOf<FloatRepr<E, M, S>> {
    using type = FloatReprDevice<E, M, S>;
};

// Convenience alias
template <typename CPUFloatRepr>
using DeviceFloatReprOfT = typename DeviceFloatReprOf<CPUFloatRepr>::type;