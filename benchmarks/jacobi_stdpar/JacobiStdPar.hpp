#pragma once

#include "definition/vector/MxVector.hpp"
#include "prelude.h"
#include <execution>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <ranges>
#include <cmath>

namespace mx::arch::gpu::stdpar {

/**
 * Helper function to convert 2D (i, j) coordinates to linear index (row-major)
 */
inline uint32_t CoordsToLinear(uint32_t i, uint32_t j, uint32_t N) {
    return i * N + j;
}

/**
 * Helper function to split linear index into 2D coordinates (row-major)
 */
inline std::pair<uint32_t, uint32_t> LinearToCoords(uint32_t idx, uint32_t N) {
    return {idx / N, idx % N};
}

/**
 * Jacobi 2D stencil computation using stdpar on plain vectors.
 * Optimized version using std::ranges::views::iota for parallel index generation.
 * 
 * Computes: dst[center] = 0.2 * (src[center] + src[left] + src[right] + src[down] + src[up])
 * 
 * @param src Source vector (read-only)
 * @param dst Destination vector (written to)
 * @param N Grid size (N x N)
 */
inline void JacobiStencilStepStdPar(const std::vector<f64>& src, std::vector<f64>& dst, uint32_t N) {
    // Interior points: i in [1, N-2], j in [1, N-2]
    // Total interior points: (N-2) * (N-2)
    // We'll iterate over interior point indices (0 to num_interior-1) and convert to (i,j)
    const uint32_t num_interior = (N - 2) * (N - 2);
    
    // Use std::ranges::views::iota to generate interior point indices in parallel
    // Then transform to actual (i,j) coordinates
    auto interior_point_indices = std::views::iota(0u, num_interior);
    
    // Use std::for_each with par_unseq to parallelize on GPU
    // This will be offloaded to GPU when compiled with nvc++ -stdpar
    std::for_each(
        std::execution::par_unseq,
        interior_point_indices.begin(),
        interior_point_indices.end(),
        [&src, &dst, N](uint32_t interior_idx) {
            // Convert interior point index to 2D coordinates
            // interior_idx ranges from 0 to (N-2)*(N-2)-1
            // Map to i in [1, N-2] and j in [1, N-2]
            uint32_t i = 1 + (interior_idx / (N - 2));
            uint32_t j = 1 + (interior_idx % (N - 2));
            
            // Compute linear index from 2D coordinates
            uint32_t center = i * N + j;
            uint32_t left = i * N + (j - 1);
            uint32_t right = i * N + (j + 1);
            uint32_t up = (i - 1) * N + j;
            uint32_t down = (i + 1) * N + j;
            
            // Compute stencil: 0.2 * (center + left + right + down + up)
            dst[center] = 0.2 * (
                src[center] +
                src[left] +
                src[right] +
                src[down] +
                src[up]
            );
        }
    );
}


/**
 * Jacobi 2D stencil computation using stdpar with on-the-fly block spreading.
 * This version works directly with MxVector blocks, spreading them on-demand.
 * 
 * @tparam MxVectorT The MxVector type
 * @param src_blocks Source MxVector blocks
 * @param dst_blocks Destination MxVector blocks (will be modified)
 * @param N Grid size (N x N)
 * @param elems_per_block Number of elements per block
 */
template <typename MxVectorT>
void JacobiStencilStepStdParOnTheFly(
    const std::vector<typename MxVectorT::BlockType>& src_blocks,
    std::vector<typename MxVectorT::BlockType>& dst_blocks,
    uint32_t N,
    size_t elems_per_block) {
    
    // Interior points: i in [1, N-2], j in [1, N-2]
    // Total interior points: (N-2) * (N-2)
    const uint32_t num_interior = (N - 2) * (N - 2);
    
    // Use std::ranges::views::iota to generate interior point indices in parallel
    // Then transform to actual (i,j) coordinates
    auto interior_point_indices = std::views::iota(0u, num_interior);
    
    // Use std::for_each with par_unseq to parallelize on GPU
    std::for_each(
        std::execution::par_unseq,
        interior_point_indices.begin(),
        interior_point_indices.end(),
        [&src_blocks, &dst_blocks, N, elems_per_block](uint32_t interior_idx) {
            // Convert interior point index to 2D coordinates
            // interior_idx ranges from 0 to (N-2)*(N-2)-1
            // Map to i in [1, N-2] and j in [1, N-2]
            uint32_t i = 1 + (interior_idx / (N - 2));
            uint32_t j = 1 + (interior_idx % (N - 2));
            
            // Compute linear index from 2D coordinates
            uint32_t center = i * N + j;
            uint32_t left = i * N + (j - 1);
            uint32_t right = i * N + (j + 1);
            uint32_t up = (i - 1) * N + j;
            uint32_t down = (i + 1) * N + j;
            
            // Spread blocks on-the-fly for the needed elements
            // Get block indices for each neighbor
            auto get_value = [&src_blocks, elems_per_block](uint32_t linear_idx) -> f64 {
                size_t block_idx = linear_idx / elems_per_block;
                size_t elem_idx = linear_idx % elems_per_block;
                return src_blocks[block_idx].RealizeAtUnsafe(elem_idx);

            };
            
            // Compute stencil: 0.2 * (center + left + right + down + up)
            f64 result = 0.2 * (
                get_value(center) +
                get_value(left) +
                get_value(right) +
                get_value(down) +
                get_value(up)
            );
            
            // Write result back to destination block
            size_t dst_block_idx = center / elems_per_block;
            size_t dst_elem_idx = center % elems_per_block;
            if (dst_block_idx < dst_blocks.size()) {
                dst_blocks[dst_block_idx].SetItemAtUnsafe(dst_elem_idx, result);
            }
        }
    );
}

/**
 * Jacobi 2D solver using C++ standard parallelism (stdpar) with GPU offloading.
 * Optimized version with parallel block spreading and on-the-fly access.
 * 
 * This implementation uses std::execution::par_unseq to offload computation
 * to GPU when compiled with nvc++ -stdpar.
 * 
 * Implementation approach:
 * 1. Spreads blocks in parallel to temporary buffers (not sequentially)
 * 2. Runs Jacobi iterations using std::for_each with par_unseq on flat vectors
 * 3. Packs results back to MxVector in parallel
 * 
 * @tparam MxVectorT The MxVector type to use
 * @param A Input/output MxVector (N*N elements, linearized row-major)
 * @param B Temporary MxVector (N*N elements, linearized row-major)
 * @param N Grid size (N x N)
 * @param steps Number of Jacobi iterations
 */
template <typename MxVectorT>
void Jacobi2DStdParSpreadOnce(MxVectorT& A, MxVectorT& B, uint32_t N, uint32_t steps) {
    // Verify sizes
    if (A.Size() != N * N || B.Size() != N * N) {
        return; // Error: size mismatch
    }
    
    // Get blocks directly - MxVector is now CPU-only, no synchronization needed
    const auto& a_blocks = A.getBlocks();
    const auto& b_blocks = B.getBlocks();
    
    // Spread MxVectors to plain vectors for stdpar GPU offloading
    // Use parallel block spreading instead of sequential
    std::vector<f64> a_flat(N * N);
    std::vector<f64> b_flat(N * N);
    
    const size_t num_blocks = a_blocks.size();
    const size_t elems_per_block = A.NumBlockElements();
    
    // Spread blocks in parallel using stdpar (much faster than sequential)
    auto block_indices = std::views::iota(size_t(0), num_blocks);
    std::for_each(
        std::execution::par_unseq,
        block_indices.begin(),
        block_indices.end(),
        [&a_blocks, &b_blocks, &a_flat, &b_flat, elems_per_block](size_t blockId) {
            // Spread block to get all elements at once
            auto a_block_data = a_blocks[blockId].Spread();
            auto b_block_data = b_blocks[blockId].Spread();
            
            size_t start = blockId * elems_per_block;
            size_t end = std::min(start + elems_per_block, a_flat.size());
            
            // Copy block data to flat array
            for (size_t i = 0; i < (end - start); ++i) {
                a_flat[start + i] = a_block_data[i];
                b_flat[start + i] = b_block_data[i];
            }
        }
    );
    
    // Run Jacobi iterations on plain vectors using stdpar
    // Each step does 2 half-steps: A->B, then B->A
    for (uint32_t t = 0; t < steps; ++t) {
        // First half-step: compute B from A
        JacobiStencilStepStdPar(a_flat, b_flat, N);
        
        // Second half-step: compute A from B (swap arguments)
        JacobiStencilStepStdPar(b_flat, a_flat, N);
    }
    
    // Pack results back to MxVector by reconstructing from flat vectors
    // The MxVector constructor uses OpenMP for parallel quantization, which is efficient
    A = MxVectorT(a_flat);
    B = MxVectorT(b_flat);
}

/**
 * Jacobi 2D solver using C++ standard parallelism (stdpar) with on-the-fly block operations.
 * This version uses JacobiStencilStepStdParOnTheFly which operates directly on MxVector blocks,
 * dequantizing values on-demand during the stencil computation.
 * 
 * This avoids allocating large intermediate float arrays by working directly with the
 * quantized block format throughout the computation.
 * 
 * @tparam MxVectorT The MxVector type to use
 * @param A Input/output MxVector (N*N elements)
 * @param B Temporary MxVector (N*N elements)
 * @param N Grid size (N x N)
 * @param steps Number of Jacobi iterations
 */
template <typename MxVectorT>
void Jacobi2DStdParOnTheFly(MxVectorT& A, MxVectorT& B, uint32_t N, uint32_t steps) {
    if (A.Size() != N * N || B.Size() != N * N) {
        return;
    }

    // Ensure B starts with A's boundary values and same quantization state
    B = A;

    const size_t elems_per_block = A.NumBlockElements();

    for (uint32_t t = 0; t < steps; ++t) {
        // Get references to blocks (will be updated in-place)
        auto& a_blocks = A.getBlocks();
        auto& b_blocks = B.getBlocks();
        // First half-step: compute B from A using on-the-fly block operations
        JacobiStencilStepStdParOnTheFly<MxVectorT>(a_blocks, b_blocks, N, elems_per_block);
        // Second half-step: compute A from B using on-the-fly block operations
        JacobiStencilStepStdParOnTheFly<MxVectorT>(b_blocks, a_blocks, N, elems_per_block);
    }
}

} // namespace mx::arch::gpu::stdpar

