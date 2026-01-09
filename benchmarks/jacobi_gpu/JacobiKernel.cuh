#pragma once

#include "arch/gpu/common.cuh"
#include "arch/gpu/SpreadKernel.cuh"
#include "arch/gpu/PackKernel.cuh"
#include "PackKernelRequant.cuh"
#include "arch/gpu/mxvector/GPUArithmetic.cuh"
#include "arch/gpu/StreamPool.h"

// Device kernel (implemented in JacobiKernel.cu)
__global__ void Jacobi2DUpdateKernel(const ElemType* __restrict__ A,
                                     ElemType* __restrict__ B,
                                     uint32_t N);

// Naive Jacobi 2D over a flattened N×N grid stored in an MxVector.
// Keeps boundary cells unchanged. Returns the final A buffer after `steps`.
// Template definition is in JacobiKernel.cu (needs to be compiled with nvcc due to kernel launches)
template <typename MxVectorT>
MxVectorT Jacobi2DGPUMxVectorNaive(const MxVectorT& A,
                                   const MxVectorT& B,
                                   uint32_t N,
                                   uint32_t steps);

// Block-wise Jacobi 2D without requantization (keeps source scalar)
// Dequantizes on-the-fly and packs using source block scalar.
// Lower overhead but scalars don't adapt to changing data distribution.
template <typename MxVectorT>
MxVectorT Jacobi2DGPUMxVectorBlockWise(const MxVectorT& A,
                                        const MxVectorT& B,
                                        uint32_t N,
                                        uint32_t steps);

// Block-wise Jacobi 2D with requantization after each full iteration
// Computes to temp f64 buffer, then requantizes with proper scalar recomputation.
// Higher overhead but scalars adapt to changing data distribution.
template <typename MxVectorT>
MxVectorT Jacobi2DGPUMxVectorBlockWiseRequant(const MxVectorT& A,
                                               const MxVectorT& B,
                                               uint32_t N,
                                               uint32_t steps);

// GPU implementation using normal CUDA arrays (f64/double)
// Allocates device memory, runs kernel, and returns result in host array
void Jacobi2DGPUArrayNaive(const ElemType* A_host,
                          const ElemType* B_host,
                          ElemType* result_host,
                          uint32_t N,
                          uint32_t steps);

// GPU Array context for separating allocation from computation
// Allows pre-allocation to exclude cudaMalloc/cudaFree/cudaMallocHost from timing
struct Jacobi2DGPUArrayContext {
    ElemType* d_A;
    ElemType* d_B;
    void* pinned_in;      // Pinned host buffer for input (2 * bytes)
    void* pinned_out;     // Pinned host buffer for output (bytes)
    uint32_t N;
    size_t bytes;
    cudaStream_t stream;
    bool initialized;
    
    Jacobi2DGPUArrayContext() : d_A(nullptr), d_B(nullptr), pinned_in(nullptr), pinned_out(nullptr), 
                                 N(0), bytes(0), stream(nullptr), initialized(false) {}
};

// Allocate device memory, upload data, and get stream (call once before timing)
void Jacobi2DGPUArraySetup(Jacobi2DGPUArrayContext& ctx,
                           const ElemType* A_host,
                           const ElemType* B_host,
                           uint32_t N);

// Run Jacobi iterations on pre-allocated device memory (this is what you time)
// No host-device transfers - data is already on GPU from Setup
void Jacobi2DGPUArrayCompute(Jacobi2DGPUArrayContext& ctx, uint32_t steps);

// Download results and free device memory and stream (call once after timing)
void Jacobi2DGPUArrayTeardown(Jacobi2DGPUArrayContext& ctx, ElemType* result_host);

// =============================================================================
// MxVector Context-based implementations for fair timing comparison
// These allow separating allocation overhead from compute time
// =============================================================================

// Context for MxVector Naive (Spread Once) - holds pre-allocated device memory
template <typename MxVectorT>
struct Jacobi2DMxVectorNaiveContext {
    using BlockT = typename MxVectorT::BlockType;
    using MxVectorViewT = MxVectorView<BlockT>;
    
    MxVectorViewT d_A_view;
    MxVectorViewT d_B_view;
    MxVectorViewT d_result_view;
    ElemType* d_A_flat;
    ElemType* d_B_flat;
    cudaStream_t stream;
    uint32_t N;
    uint32_t num_blocks;
    uint32_t elems_per_block;
    bool initialized;
    
    Jacobi2DMxVectorNaiveContext() : d_A_flat(nullptr), d_B_flat(nullptr), 
        stream(nullptr), N(0), num_blocks(0), elems_per_block(0), initialized(false) {}
};

// Context for MxVector BlockWise (No Requant)
template <typename MxVectorT>
struct Jacobi2DMxVectorBlockWiseContext {
    using BlockT = typename MxVectorT::BlockType;
    using MxVectorViewT = MxVectorView<BlockT>;
    
    MxVectorViewT d_A_view;
    MxVectorViewT d_B_view;
    cudaStream_t stream;
    uint32_t N;
    uint32_t num_blocks;
    uint32_t elems_per_block;
    bool initialized;
    
    Jacobi2DMxVectorBlockWiseContext() : stream(nullptr), N(0), num_blocks(0), 
        elems_per_block(0), initialized(false) {}
};

// Context for MxVector BlockWise with Requantization
template <typename MxVectorT>
struct Jacobi2DMxVectorBlockWiseRequantContext {
    using BlockT = typename MxVectorT::BlockType;
    using MxVectorViewT = MxVectorView<BlockT>;
    
    MxVectorViewT d_A_view;
    MxVectorViewT d_B_view;
    ElemType* d_temp_flat;
    cudaStream_t stream;
    uint32_t N;
    uint32_t num_blocks;
    uint32_t elems_per_block;
    bool initialized;
    
    Jacobi2DMxVectorBlockWiseRequantContext() : d_temp_flat(nullptr), stream(nullptr), 
        N(0), num_blocks(0), elems_per_block(0), initialized(false) {}
};

// Setup/Compute/Teardown for MxVector Naive (Spread Once)
template <typename MxVectorT>
void Jacobi2DMxVectorNaiveSetup(Jacobi2DMxVectorNaiveContext<MxVectorT>& ctx,
                                 const MxVectorT& A, const MxVectorT& B, uint32_t N);

template <typename MxVectorT>
void Jacobi2DMxVectorNaiveCompute(Jacobi2DMxVectorNaiveContext<MxVectorT>& ctx, uint32_t steps);

template <typename MxVectorT>
MxVectorT Jacobi2DMxVectorNaiveTeardown(Jacobi2DMxVectorNaiveContext<MxVectorT>& ctx);

// Setup/Compute/Teardown for MxVector BlockWise (No Requant)
template <typename MxVectorT>
void Jacobi2DMxVectorBlockWiseSetup(Jacobi2DMxVectorBlockWiseContext<MxVectorT>& ctx,
                                     const MxVectorT& A, const MxVectorT& B, uint32_t N);

template <typename MxVectorT>
void Jacobi2DMxVectorBlockWiseCompute(Jacobi2DMxVectorBlockWiseContext<MxVectorT>& ctx, uint32_t steps);

template <typename MxVectorT>
MxVectorT Jacobi2DMxVectorBlockWiseTeardown(Jacobi2DMxVectorBlockWiseContext<MxVectorT>& ctx);

// Setup/Compute/Teardown for MxVector BlockWise with Requantization
template <typename MxVectorT>
void Jacobi2DMxVectorBlockWiseRequantSetup(Jacobi2DMxVectorBlockWiseRequantContext<MxVectorT>& ctx,
                                            const MxVectorT& A, const MxVectorT& B, uint32_t N);

template <typename MxVectorT>
void Jacobi2DMxVectorBlockWiseRequantCompute(Jacobi2DMxVectorBlockWiseRequantContext<MxVectorT>& ctx, uint32_t steps);

template <typename MxVectorT>
MxVectorT Jacobi2DMxVectorBlockWiseRequantTeardown(Jacobi2DMxVectorBlockWiseRequantContext<MxVectorT>& ctx);

