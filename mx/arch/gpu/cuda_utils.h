#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(ans) { cudaAssert((ans), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",
                cudaGetErrorString(code), file, line);
        fflush(stderr);
        std::exit(1);
    }
}

inline void CUDA_CHECK_KERNEL(const char* msg = nullptr)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error%s%s: %s\n",
                msg ? " (" : "",
                msg ? msg : "",
                cudaGetErrorString(err));
        std::exit(1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

#endif // CUDA_UTILS_CUH