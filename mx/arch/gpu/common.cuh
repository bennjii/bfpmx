#ifndef BFPMX_GPU_COMMON_CUH
#define BFPMX_GPU_COMMON_CUH
#include "definition/alias.h"
#include <array>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using ElemType = f64;
enum class ArithmeticOp : uint8_t {
    Add,
    Sub,
    Mul,
    Div
};
#endif // BFPMX_GPU_COMMON_CUH