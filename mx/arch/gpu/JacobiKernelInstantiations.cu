// Explicit instantiations for Jacobi2DGPUMxVectorNaive
// This file contains explicit template instantiations for various block sizes
// to ensure they are compiled and available for use.

#include "JacobiKernel.cuh"
#include "definition/vector/MxVector.hpp"

// Include the template definition from JacobiKernel.cu
// We need to include the implementation to make explicit instantiation work
// Note: We skip the kernel and non-template function definitions by defining a guard
#define JACOBI_KERNEL_INSTANTIATIONS_ONLY
#include "JacobiKernel.cu"
#undef JACOBI_KERNEL_INSTANTIATIONS_ONLY

// Type aliases for different block sizes
using GPUVector4 = mx::vector::MxVector<BlockDims<4>, unsigned char, fp8::E4M3Type,
                                        GPUArithmeticNaive, MaximumFractionalQuantization>;

using GPUVector8 = mx::vector::MxVector<BlockDims<8>, unsigned char, fp8::E4M3Type,
                                        GPUArithmeticNaive, MaximumFractionalQuantization>;

using GPUVector16 = mx::vector::MxVector<BlockDims<16>, unsigned char, fp8::E4M3Type,
                                         GPUArithmeticNaive, MaximumFractionalQuantization>;

using GPUVector32 = mx::vector::MxVector<BlockDims<32>, unsigned char, fp8::E4M3Type,
                                         GPUArithmeticNaive, MaximumFractionalQuantization>;

using GPUVector64 = mx::vector::MxVector<BlockDims<64>, unsigned char, fp8::E4M3Type,
                                         GPUArithmeticNaive, MaximumFractionalQuantization>;

using GPUVector128 = mx::vector::MxVector<BlockDims<128>, unsigned char, fp8::E4M3Type,
                                          GPUArithmeticNaive, MaximumFractionalQuantization>;

// Explicit template instantiations
template GPUVector4 Jacobi2DGPUMxVectorNaive<GPUVector4>(
    const GPUVector4&, const GPUVector4&, uint32_t, uint32_t);

template GPUVector8 Jacobi2DGPUMxVectorNaive<GPUVector8>(
    const GPUVector8&, const GPUVector8&, uint32_t, uint32_t);

template GPUVector16 Jacobi2DGPUMxVectorNaive<GPUVector16>(
    const GPUVector16&, const GPUVector16&, uint32_t, uint32_t);

template GPUVector32 Jacobi2DGPUMxVectorNaive<GPUVector32>(
    const GPUVector32&, const GPUVector32&, uint32_t, uint32_t);

template GPUVector64 Jacobi2DGPUMxVectorNaive<GPUVector64>(
    const GPUVector64&, const GPUVector64&, uint32_t, uint32_t);

template GPUVector128 Jacobi2DGPUMxVectorNaive<GPUVector128>(
    const GPUVector128&, const GPUVector128&, uint32_t, uint32_t);

