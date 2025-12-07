#pragma once
#include <functional>
#include "prelude.h"
#include "omp.h"
namespace mx::vector {
    template <OneDimensionalBlockDimsType BlockShape, typename ScalarType = unsigned char,
        IFloatRepr Float=fp8::E4M3Type,
        template <typename> typename ArithmeticPolicy = CPUArithmetic,
        template <std::size_t, BlockDimsType, IFloatRepr> typename QuantizationPolicy = MaximumFractionalQuantization>
    class MxVector { 
        public:
        using BlockType =
        Block<ScalarType, BlockShape, Float, ArithmeticPolicy,           
        QuantizationPolicy>;

        MxVector(size_t num_elements) {
            size_t num_blocks = (num_elements + BlockType::NumElems - 1) / BlockType::NumElems;
            blocks_.resize(num_blocks);
            num_elements_ = num_elements;
        }

        MxVector(const std::vector<f64>& data) {
            size_t num_blocks = (data.size() + BlockType::NumElems - 1) / BlockType::NumElems;
            blocks_.resize(num_blocks);
            num_elements_ = data.size();

            #pragma omp parallel for
            for (size_t blockId = 0; blockId < num_blocks; ++blockId) {
                auto block_data = std::array<f64, BlockType::NumElems>{};
                size_t start = blockId * BlockType::NumElems;
                size_t end = std::min(start + BlockType::NumElems, data.size());
                for (size_t i = start; i < end; ++i){
                    block_data[i-start] = data[i];
                }
                blocks_[blockId] = BlockType(block_data);
            }
        }

        MxVector(const std::vector<BlockType>& blocks, size_t num_elements)
            : blocks_(blocks), num_elements_(num_elements) {}

        f64 ItemAt(size_t index) const {
            size_t block_index = index / BlockType::NumElems;
            size_t elem_index = index % BlockType::NumElems;
            auto result = blocks_[block_index].RealizeAt(elem_index);
            return result.value_or(0.0);  // Unwrap optional, default to 0.0 if empty
        }

        const BlockType& BlockAt(size_t block_index) const noexcept{
            return blocks_[block_index];
        }

        bool SetItemAt(size_t index, f64 value) {
            size_t block_index = index / BlockType::NumElems;
            if (block_index >= blocks_.size()) {
                return false;
            }
            size_t elem_index = index % BlockType::NumElems;
            blocks_[block_index].SetItemAtUnsafe(elem_index, value);
            return true;
        }

        bool SetBlockAt(size_t block_index, const BlockType& block) {
            if (block_index >= blocks_.size()) {
                return false;
            }
            blocks_[block_index] = block;
            return true;
        }

        size_t Size() const {
            return num_elements_;
        }

        // For GPU usage, we should change it for security/abstraction later
        std::vector<BlockType> getBlocks() const {
            return blocks_;
        }

        size_t NumBlocks() const {
            return blocks_.size();
        }

        size_t SizeInBytes() const {
            return blocks_.size() * sizeof(BlockType);
        }

        size_t NumBlockElements() const {
            return BlockType::NumElems;
        }

        void asString() const {
            for (size_t i = 0; i < blocks_.size(); ++i) {
                std::cout << "Block " << i << ":\n" << blocks_[i].asString() << "\n";
            }
        }

        private: 
        std::vector<BlockType> blocks_;
        size_t num_elements_;
        
    };
} // namespace mx