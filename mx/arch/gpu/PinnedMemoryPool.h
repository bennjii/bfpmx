#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <mutex>
#include <queue>
#include <memory>
#include <cstddef>
#include "cuda_utils.h"

/**
 * Thread-safe pool of pinned memory buffers for reuse.
 * Reduces overhead from allocating/deallocating pinned memory per operation.
 */
class PinnedMemoryPool {
public:
    struct Buffer {
        void* ptr;
        size_t size;
        
        Buffer(void* p, size_t s) : ptr(p), size(s) {}
        
        ~Buffer() {
            if (ptr) {
                CUDA_CHECK(cudaFreeHost(ptr));
            }
        }
        
        // Non-copyable, movable
        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;
        Buffer(Buffer&& other) noexcept : ptr(other.ptr), size(other.size) {
            other.ptr = nullptr;
            other.size = 0;
        }
    };

    PinnedMemoryPool(size_t max_buffers = 8) : max_buffers_(max_buffers) {}

    ~PinnedMemoryPool() {
        std::lock_guard<std::mutex> lock(mutex_);
        buffers_.clear();
    }

    // Non-copyable
    PinnedMemoryPool(const PinnedMemoryPool&) = delete;
    PinnedMemoryPool& operator=(const PinnedMemoryPool&) = delete;

    /**
     * Acquire a pinned memory buffer of at least the requested size.
     * Returns a buffer that can be larger than requested.
     * If no suitable buffer is available, allocates a new one.
     */
    std::unique_ptr<Buffer> Acquire(size_t min_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Try to find a buffer that's large enough
        for (auto it = buffers_.begin(); it != buffers_.end(); ++it) {
            if ((*it)->size >= min_size) {
                auto buffer = std::move(*it);
                buffers_.erase(it);
                return buffer;
            }
        }
        
        // No suitable buffer found, allocate a new one
        // Round up to next 256-byte boundary for alignment
        size_t aligned_size = (min_size + 255) & ~255;
        void* ptr = nullptr;
        CUDA_CHECK(cudaMallocHost(&ptr, aligned_size));
        return std::make_unique<Buffer>(ptr, aligned_size);
    }

    /**
     * Release a buffer back to the pool.
     * Only keeps buffers up to max_buffers_ limit.
     */
    void Release(std::unique_ptr<Buffer> buffer) {
        if (!buffer || !buffer->ptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Only keep buffers if we're under the limit
        if (buffers_.size() < max_buffers_) {
            buffers_.push_back(std::move(buffer));
        }
        // Otherwise, buffer is automatically freed when unique_ptr goes out of scope
    }

    /**
     * Clear all buffers from the pool.
     */
    void Clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        buffers_.clear();
    }

    size_t PoolSize() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffers_.size();
    }

private:
    size_t max_buffers_;
    std::vector<std::unique_ptr<Buffer>> buffers_;
    mutable std::mutex mutex_;
};

/**
 * Global pinned memory pool instance (thread-safe singleton).
 * Use this for most operations.
 */
inline PinnedMemoryPool& GetGlobalPinnedMemoryPool() {
    static PinnedMemoryPool pool(8);  // Default max buffers of 8
    return pool;
}
