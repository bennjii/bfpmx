//
// Created by Benjamin White on 10/10/2025.
//

#ifndef BFPMX_ARCH_PRELUDE_H
#define BFPMX_ARCH_PRELUDE_H

#pragma once

#include "Arithmetic.h"

#include "cpu/CPUArithmetic.h"
#include "cpu/CPUArithmeticWithoutMarshalling.h"

#ifdef HAS_CUDA
#include "gpu/GPUArithmetic.cuh"
#endif

// #if defined(__x86_64__) || defined(_M_X64)
// #include "cpu/x86_64.cpp"
// #elif defined(__aarch64__) || defined(_M_ARM64)
// #include "cpu/aarch64.cpp"
// #endif

#endif // BFPMX_ARCH_PRELUDE_H