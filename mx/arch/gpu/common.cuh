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

template <uint8_t Exp, uint8_t Significand, uint8_t Sign>
struct BlockView {
    static constexpr uint8_t ExponentBits = Exp;
    static constexpr uint8_t SignificandBits = Significand;
    static constexpr uint8_t SignBits = Sign;

    const uint8_t* data;      // pointer to packed quantized bytes (GPU memory)
    uint8_t scalar;    // pointer to scalar exponent bytes (GPU memory)
    uint32_t num_elems;            // number of packed elements
    uint8_t elem_size_bytes;      // bytes per element (FloatRepr::SizeBytes())
};

#endif // BFPMX_GPU_COMMON_CUH