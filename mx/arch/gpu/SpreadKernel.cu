#include "SpreadKernel.cuh"
#include "cuda_utils.h"
// ======================================================
// Kernel definition
// ======================================================
template <typename FloatD, typename BlockViewT>
__global__ void SpreadKernel(const BlockViewT* left_view,
                             const BlockViewT* right_view,
                             ElemType* d_l,
                             ElemType* d_r)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < left_view->num_elems) {
        d_l[idx] = FloatD::UnmarshalDevice(left_view->data + idx * left_view->elem_size_bytes) * left_view->scalar;
        d_r[idx] = FloatD::UnmarshalDevice(right_view->data + idx * right_view->elem_size_bytes) * right_view->scalar;
    }
}

// ======================================================
// Host launcher definition
// ======================================================
template <typename BlockViewT>
void LaunchSpreadKernel(const BlockViewT* d_lhs,
                        const BlockViewT* d_rhs,
                        ElemType* d_l,
                        ElemType* d_r)
{
    const int blockSize = 256;
    const int numBlocks = (d_lhs->num_elems + blockSize - 1) / blockSize;

    using BV = BlockViewT;
    using FloatTypeD =
        FloatReprDevice<BV::ExponentBits, BV::SignificandBits, BV::SignBits>;

    SpreadKernel<FloatTypeD, BlockViewT>
        <<<numBlocks, blockSize>>>(d_lhs, d_rhs, d_l, d_r);

    CUDA_CHECK_KERNEL();
}

// ======================================================
// Explicit instantiations
// ======================================================

// Block types you want to support:
using BV431 = BlockView<4,3,1>;
using BV521 = BlockView<5,2,1>;

// ---- Instantiate kernels ----
template __global__
void SpreadKernel<FloatReprDevice<4,3,1>, BV431>(
    const BV431*, const BV431*, ElemType*, ElemType*);

template __global__
void SpreadKernel<FloatReprDevice<5,2,1>, BV521>(
    const BV521*, const BV521*, ElemType*, ElemType*);

// ---- Instantiate host wrappers ----
template
void LaunchSpreadKernel<BV431>(
    const BV431*, const BV431*, ElemType*, ElemType*);

template
void LaunchSpreadKernel<BV521>(
    const BV521*, const BV521*, ElemType*, ElemType*);