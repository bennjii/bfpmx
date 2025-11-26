// This header file must NOT be included in any non-CUDA file, even with a HAS_CUDA flag
#ifndef BFPMX_ARCH_PRELUDE_GPU_H
#define BFPMX_ARCH_PRELUDE_GPU_H

#ifdef HAS_CUDA
#include "ArithmeticKernels.cuh"
#include "common.cuh"
#include "GPUArithmetic.cuh"
#include "SpreadKernel.cuh"
#endif

#endif // BFPMX_ARCH_PRELUDE_H