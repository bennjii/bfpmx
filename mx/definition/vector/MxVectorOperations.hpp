#pragma once
#include "definition/vector/MxVector.hpp"
#include <optional>
#include <omp.h>

namespace mx::vector::ops {
    using ElemType = f64;
    // For now we do not check the dimension of the vectors,
    // as the dimension is dynamic and runtime check comes with 
    // performance cost. If the type are mismatched, it results in a
    // undefined behavior.
    template <typename T, typename OutputType = f64>
    OutputType Dot(const T& a, const T& b) {
        OutputType result = 0.;
        for (auto i = 0; i < a.NumBlocks(); ++i) {
            const auto a_block = a.BlockAt(i);
            const auto b_block = b.BlockAt(i);
            result += a_block * b_block;
        }
        return result;
    }

    template <typename T>
    T AddBlockwise(const T& a, const T& b) {
        std::vector<typename T::BlockType> result_blocks(a.NumBlocks());
        #pragma omp parallel for
        for (auto i = 0; i < a.NumBlocks(); ++i) {
            const auto a_block = a.BlockAt(i);
            const auto b_block = b.BlockAt(i);
            result_blocks[i] = a_block + b_block;
        }
        return T(result_blocks, a.Size());
    }

    template <typename T>
    T AddPointwiseGPU(const T& a, const T& b) {
        return AddPointwiseGPUMxVector(a, b);
    }
} // namespace mx

