# Collect all CUDA kernel source files
set(GPU_KERNEL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/ArithmeticKernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/SpreadKernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/PackKernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/FusedKernel.cu
)

# Compile kernels as an OBJECT library (device + host code)
add_library(bfpmx_cuda_kernels STATIC ${GPU_KERNEL_SOURCES})

# NVCC compile settings
set_target_properties(bfpmx_cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
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

target_link_libraries(bfpmx_gpu INTERFACE
    bfpmx_cuda_kernels
    CUDA::cudart
)

# NVTX support for profiling
# Fetch NVTX from GitHub and use its CMake targets
if(CMAKE_CUDA_COMPILER)
    include(FetchContent)
    # Use the same base directory as other dependencies
    set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/.deps")
    FetchContent_Declare(
        nvtx
        GIT_REPOSITORY https://github.com/NVIDIA/NVTX.git
        GIT_TAG        v3.1.1
    )
    
    FetchContent_MakeAvailable(nvtx)
    
    # Add NVTX c subdirectory to get nvtx3-c target
    # Use a unique binary directory name to avoid conflicts
    add_subdirectory(${nvtx_SOURCE_DIR}/c ${nvtx_BINARY_DIR}/c EXCLUDE_FROM_ALL)
    
    # Link against nvtx3-c target (provides nvToolsExt.h compatibility via nvtx3/nvToolsExt.h)
    target_link_libraries(bfpmx_cuda_kernels PRIVATE nvtx3-c)
    message(STATUS "Using NVTX from GitHub: ${nvtx_SOURCE_DIR}")
endif()