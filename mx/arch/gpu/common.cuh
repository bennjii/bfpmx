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

struct BlockView {
    const uint8_t* data;      // pointer to packed quantized bytes (GPU memory)
    const uint8_t* scalar;    // pointer to scalar exponent bytes (GPU memory)
    int num_elems;            // number of packed elements
    int elem_size_bytes;      // bytes per element (FloatRepr::SizeBytes())
    int scalar_size_bytes;    // bytes for scalar exponent
};
#endif // BFPMX_GPU_COMMON_CUH