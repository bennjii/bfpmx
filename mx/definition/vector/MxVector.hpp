#pragma once
#include <functional>
#include "definition/block_float/block/Block.h"
#include "definition/block_float/block/BlockDims.h"
#include "definition/block_float/repr/FloatRepr.h"
#include "definition/quantization/MaximumFractionalQuantization.h"

namespace mx::vector {
    template <OneDimensionalBlockDimsType BlockShape, size_t ScalarSizeBytes=1,
        IFloatRepr Float=fp8::E4M3Type,
        template <typename> typename ArithmeticPolicy = CPUArithmetic,
        template <std::size_t, BlockDimsType, 
            IFloatRepr, template <typename> typename ArithmeticPolicy_> typename QuantizationPolicy = MaximumFractionalQuantization>
    class MxVector { 
        public:
        using BlockType =
        Block<ScalarSizeBytes, BlockShape, Float, ArithmeticPolicy,           
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

            for (size_t blockId = 0; blockId < data.size(); ++blockId) {
                size_t block_index = blockId / BlockType::NumElems;
                size_t elem_index = blockId % BlockType::NumElems;
                auto block_data = std::array<f64, BlockType::NumElems>{};
                for (size_t i = 0; i < BlockType::NumElems; ++i){
                    if(size_t idx = block_index * BlockType::NumElems + i; idx < data.size())
                        block_data[i] = data[idx];
                    else
                        block_data[i] = 0.0;
                }
                blocks_[block_index] = BlockType(block_data);
            }
        }

        MxVector(const std::vector<BlockType>& blocks, size_t num_elements)
            : blocks_(blocks), num_elements_(num_elements) {}

        std::optional<f64> ItemAt(size_t index) const {
            if (index >= num_elements_) {
                return std::nullopt;
            }
            size_t block_index = index / BlockType::NumElems;
            size_t elem_index = index % BlockType::NumElems;
            return blocks_[block_index].RealizeAt(elem_index);
        }

        std::optional<BlockType> BlockAt(size_t block_index) const {
            if (block_index >= blocks_.size()) {
                return std::nullopt;
            }
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

        size_t NumBlocks() const {
            return blocks_.size();
        }

        size_t SizeInBytes() const {
            return blocks_.size() * sizeof(BlockType);
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