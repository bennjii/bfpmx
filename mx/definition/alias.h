//
// Created by Benjamin White on 10/10/2025.
//

#ifndef BFPMX_ALIAS_H
#define BFPMX_ALIAS_H

#include <cstdint>

// Float Aliases
typedef float f32;
typedef double f64;

// Integer Aliases
typedef int8_t i8;
typedef uint8_t u8;

typedef int16_t i16;
typedef uint16_t u16;

typedef int32_t i32;
typedef uint32_t u32;

typedef int64_t i64;
typedef uint64_t u64;

// Macro definitions for CUDA compatibility
#ifdef HAS_CUDA
    #include "cuda_runtime.h"
    #include "device_launch_parameters.h"
    #define HD __host__ __device__
#else
    #define HD
#endif

#endif
