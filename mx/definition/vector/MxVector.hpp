#pragma once
#include <functional>
#include <memory>
#include "prelude.h"
#include "omp.h"
#include "definition/vector/DataLocation.h"
#ifdef HAS_CUDA
#include "arch/gpu/mxvector/GPUArithmetic.cuh"
#endif

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
            data_location_ = DataLocation::CPU_ONLY;
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
            data_location_ = DataLocation::CPU_ONLY;
        }

        MxVector(const std::vector<BlockType>& blocks, size_t num_elements)
            : blocks_(blocks), num_elements_(num_elements), data_location_(DataLocation::CPU_ONLY) {}

        ~MxVector() {
            #ifdef HAS_CUDA
            if (gpu_view_ && (data_location_ == DataLocation::GPU_ONLY || data_location_ == DataLocation::BOTH)) {
                FreeDeviceMxVectorView(gpu_view_.get());
            }
            #endif
        }

        MxVector(const MxVector& other) 
            : blocks_(other.blocks_), num_elements_(other.num_elements_), data_location_(DataLocation::CPU_ONLY) {
            #ifdef HAS_CUDA
            if (other.data_location_ == DataLocation::GPU_ONLY || other.data_location_ == DataLocation::BOTH) {
                ensureGPUData();
            }
            #endif
        }

        MxVector& operator=(const MxVector& other) {
            if (this != &other) {
                #ifdef HAS_CUDA
                if (gpu_view_ && (data_location_ == DataLocation::GPU_ONLY || data_location_ == DataLocation::BOTH)) {
                    FreeDeviceMxVectorView(gpu_view_.get());
                }
                #endif
                blocks_ = other.blocks_;
                num_elements_ = other.num_elements_;
                data_location_ = DataLocation::CPU_ONLY;
                #ifdef HAS_CUDA
                if (other.data_location_ == DataLocation::GPU_ONLY || other.data_location_ == DataLocation::BOTH) {
                    ensureGPUData();
                }
                #endif
            }
            return *this;
        }

        MxVector(MxVector&& other) noexcept
            : blocks_(std::move(other.blocks_)), 
              num_elements_(other.num_elements_),
              data_location_(other.data_location_),
              gpu_view_(std::move(other.gpu_view_)) {
            other.data_location_ = DataLocation::INVALID;
        }

        MxVector& operator=(MxVector&& other) noexcept {
            if (this != &other) {
                #ifdef HAS_CUDA
                if (gpu_view_ && (data_location_ == DataLocation::GPU_ONLY || data_location_ == DataLocation::BOTH)) {
                    FreeDeviceMxVectorView(gpu_view_.get());
                }
                #endif
                blocks_ = std::move(other.blocks_);
                num_elements_ = other.num_elements_;
                data_location_ = other.data_location_;
                gpu_view_ = std::move(other.gpu_view_);
                other.data_location_ = DataLocation::INVALID;
            }
            return *this;
        }

        f64 ItemAt(size_t index) const {
            ensureCPUData();
            size_t block_index = index / BlockType::NumElems;
            size_t elem_index = index % BlockType::NumElems;
            auto result = blocks_[block_index].RealizeAt(elem_index);
            return result.value_or(0.0);
        }

        const BlockType& BlockAt(size_t block_index) const noexcept{
            ensureCPUData();
            return blocks_[block_index];
        }

        bool SetItemAt(size_t index, f64 value) {
            ensureCPUData();
            size_t block_index = index / BlockType::NumElems;
            if (block_index >= blocks_.size()) {
                return false;
            }
            size_t elem_index = index % BlockType::NumElems;
            blocks_[block_index].SetItemAtUnsafe(elem_index, value);
            #ifdef HAS_CUDA
            if (data_location_ == DataLocation::BOTH || data_location_ == DataLocation::GPU_ONLY) {
                data_location_ = DataLocation::CPU_ONLY;
                if (gpu_view_) {
                    FreeDeviceMxVectorView(gpu_view_.get());
                    gpu_view_.reset();
                }
            }
            #endif
            return true;
        }

        bool SetBlockAt(size_t block_index, const BlockType& block) {
            ensureCPUData();
            if (block_index >= blocks_.size()) {
                return false;
            }
            blocks_[block_index] = block;
            #ifdef HAS_CUDA
            if (data_location_ == DataLocation::BOTH || data_location_ == DataLocation::GPU_ONLY) {
                data_location_ = DataLocation::CPU_ONLY;
                if (gpu_view_) {
                    FreeDeviceMxVectorView(gpu_view_.get());
                    gpu_view_.reset();
                }
            }
            #endif
            return true;
        }

        size_t Size() const {
            return num_elements_;
        }

        std::vector<BlockType> getBlocks() const {
            ensureCPUData();
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
            ensureCPUData();
            for (size_t i = 0; i < blocks_.size(); ++i) {
                std::cout << "Block " << i << ":\n" << blocks_[i].asString() << "\n";
            }
        }

        #ifdef HAS_CUDA
        using MxVectorViewT = MxVectorView<BlockType>;
        
        void ensureCPUData() const {
            if (data_location_ == DataLocation::GPU_ONLY && gpu_view_) {
                auto host_vector = ToHostMxVector<BlockType, MxVector>(*gpu_view_);
                const_cast<MxVector*>(this)->blocks_ = host_vector.getBlocks();
                const_cast<MxVector*>(this)->data_location_ = DataLocation::BOTH;
            }
        }

        void ensureGPUData() const {
            if (data_location_ == DataLocation::CPU_ONLY || data_location_ == DataLocation::INVALID) {
                if (!gpu_view_) {
                    gpu_view_ = std::make_unique<MxVectorViewT>(ToDeviceMxVectorView(const_cast<MxVector&>(*this)));
                }
                data_location_ = (data_location_ == DataLocation::CPU_ONLY) ? DataLocation::BOTH : DataLocation::GPU_ONLY;
            }
        }

        MxVectorViewT& getGPUView() {
            ensureGPUData();
            return *gpu_view_;
        }

        const MxVectorViewT& getGPUView() const {
            ensureGPUData();
            return *gpu_view_;
        }

        DataLocation getDataLocation() const {
            return data_location_;
        }

        void setGPUView(MxVectorViewT view) {
            if (gpu_view_ && (data_location_ == DataLocation::GPU_ONLY || data_location_ == DataLocation::BOTH)) {
                FreeDeviceMxVectorView(gpu_view_.get());
            }
            gpu_view_ = std::make_unique<MxVectorViewT>(view);
            data_location_ = DataLocation::GPU_ONLY;
        }
        #endif

        private: 
        std::vector<BlockType> blocks_;
        size_t num_elements_;
        mutable DataLocation data_location_;
        #ifdef HAS_CUDA
        mutable std::unique_ptr<MxVectorViewT> gpu_view_;
        #endif
        
    };
} // namespace mx