#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <mutex>
#include <queue>
#include <condition_variable>
#include "cuda_utils.h"

/**
 * Thread-safe pool of CUDA streams for reuse.
 * Reduces overhead from creating/destroying streams per operation.
 */
class StreamPool {
public:
    StreamPool(size_t pool_size = 4) : pool_size_(pool_size) {
        streams_.reserve(pool_size_);
        for (size_t i = 0; i < pool_size_; ++i) {
            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));
            streams_.push_back(stream);
            available_streams_.push(i);
        }
    }

    ~StreamPool() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (cudaStream_t stream : streams_) {
            CUDA_CHECK(cudaStreamDestroy(stream));
        }
    }

    // Non-copyable
    StreamPool(const StreamPool&) = delete;
    StreamPool& operator=(const StreamPool&) = delete;

    /**
     * Acquire a stream from the pool.
     * Blocks if no streams are available.
     */
    cudaStream_t Acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (available_streams_.empty()) {
            // Wait for a stream to become available
            cv_.wait(lock);
        }
        size_t idx = available_streams_.front();
        available_streams_.pop();
        return streams_[idx];
    }

    /**
     * Release a stream back to the pool.
     */
    void Release(cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mutex_);
        // Find the index of the stream
        for (size_t i = 0; i < streams_.size(); ++i) {
            if (streams_[i] == stream) {
                available_streams_.push(i);
                cv_.notify_one();
                return;
            }
        }
        // Stream not found in pool - should not happen
        // But we'll handle it gracefully
    }

    /**
     * Synchronize all streams in the pool.
     */
    void SynchronizeAll() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (cudaStream_t stream : streams_) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }

    size_t PoolSize() const { return pool_size_; }

private:
    size_t pool_size_;
    std::vector<cudaStream_t> streams_;
    std::queue<size_t> available_streams_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

/**
 * Global stream pool instance (thread-safe singleton).
 * Use this for most operations.
 */
inline StreamPool& GetGlobalStreamPool() {
    static StreamPool pool(4);  // Default pool size of 4
    return pool;
}
