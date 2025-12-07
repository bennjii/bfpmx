#include "cuda_utils.h"
#include "PackKernel.cuh"  // for ElemType

// Device kernel: pack flat ElemType -> BlockView::data
template <typename FloatD, typename BlockViewT>
__global__ void PackKernel(const ElemType* d_in,
                           BlockViewT* out_block)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= out_block->num_elems) return;

    // Convert ElemType (float) to target packed bytes
    double val = d_in[idx];

    // Marshal value into target layout bytes
    auto packed_bytes = FloatD::MarshalDevice(val);

    // Copy bytes into device memory
    #pragma unroll
    for (unsigned int b = 0; b < FloatD::SizeBytes(); ++b) {
        out_block->data[idx * FloatD::SizeBytes() + b] = packed_bytes[b];
    }
}

template <typename BlockViewT>
void LaunchPackKernel(const ElemType* d_in,
                      BlockViewT* out_block)
{
    const int blockSize = 256;
    const int numBlocks = (out_block->num_elems + blockSize - 1) / blockSize;
    
    using BV = BlockViewT;
    using FloatTypeD =
        FloatReprDevice<BV::ExponentBits, BV::SignificandBits, BV::SignBits>;

    PackKernel<FloatTypeD, BlockViewT>
        <<<numBlocks, blockSize>>>(d_in, out_block);
    CUDA_CHECK_KERNEL();
}

// Block types you want to support:
using BV431 = BlockView<4,3,1>;
using BV521 = BlockView<5,2,1>;

// ---- Instantiate kernels ----
template __global__
void PackKernel<FloatReprDevice<4,3,1>, BV431>(
    const ElemType*, BV431*);

template __global__
void PackKernel<FloatReprDevice<5,2,1>, BV521>(
    const ElemType*, BV521*);

// ---- Instantiate host wrappers ----
template
void LaunchPackKernel<BV431>(
    const ElemType*, BV431*);

template
void LaunchPackKernel<BV521>(
    const ElemType*, BV521*);