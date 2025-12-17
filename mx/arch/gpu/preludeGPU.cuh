// This header file must NOT be included in any non-CUDA file, even with a HAS_CUDA flag
#ifndef BFPMX_ARCH_PRELUDE_GPU_H
#define BFPMX_ARCH_PRELUDE_GPU_H

#ifdef HAS_CUDA
#include "ArithmeticKernels.cuh"
#include "common.cuh"
#include "GPUArithmeticBlock.cuh"
#include "SpreadKernel.cuh"
#include "PackKernel.cuh"
#include "FusedKernel.cuh"
#include "JacobiKernel.cuh"
#include "HeatKernel.cuh"
#include "mxvector/GPUArithmetic.cuh"
#endif

#endif // BFPMX_ARCH_PRELUDE_H