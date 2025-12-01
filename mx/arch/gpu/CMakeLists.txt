# Collect all CUDA kernel source files
set(GPU_KERNEL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/ArithmeticKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/SpreadKernel.cu
)

# Compile kernels as an OBJECT library (device + host code)
add_library(bfpmx_cuda_kernels OBJECT ${GPU_KERNEL_SOURCES})

# NVCC compile settings
set_target_properties(bfpmx_cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

target_include_directories(bfpmx_cuda_kernels 
    PRIVATE 
    ${CMAKE_SOURCE_DIR}/mx
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# CUDA-specific flags (optional but recommended)
target_compile_options(bfpmx_cuda_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Headers-only GPU interface library
add_library(bfpmx_gpu INTERFACE)

# GPU interface must expose its include directory
target_include_directories(bfpmx_gpu INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link all GPU kernels + CUDA runtime when user links bfpmx_gpu
find_package(CUDAToolkit REQUIRED)

target_link_libraries(bfpmx_gpu INTERFACE
    bfpmx_cuda_kernels
    CUDA::cudart
)