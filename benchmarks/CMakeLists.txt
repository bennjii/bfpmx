
include(FetchContent)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.4
)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googlebenchmark)

# Link pthread on most platforms
if(NOT WIN32)
    set(BENCHMARK_LIBS benchmark::benchmark pthread)
else()
    set(BENCHMARK_LIBS benchmark::benchmark)
endif()

add_executable(bench_vector_ops mx_vec_ops.cpp)
target_link_libraries(bench_vector_ops 
    PRIVATE 
    bfpmx 
    bfpmx_gpu
    ${BENCHMARK_LIBS} 
)

# Benchmarks #
add_executable(jacobi2d jacobi.cpp)
target_link_libraries(jacobi2d PRIVATE bfpmx)

# StdPar Benchmark (requires nvc++ compiler with -stdpar flag for GPU offloading)
# Use custom command to compile with nvc++ while rest of project uses g++
find_program(NVCXX_COMPILER nvc++ PATHS
    /usr/local/bin
    /usr/bin
    $ENV{PATH}
    $ENV{HPC_SDK}/Linux_x86_64/*/compilers/bin
    /home/mipham/scratch/hpc_sdk/Linux_x86_64/25.9/compilers/bin
)

if(NVCXX_COMPILER AND CMAKE_CUDA_COMPILER)
    # Get paths for compilation
    get_target_property(BFPMX_LIB_PATH bfpmx ARCHIVE_OUTPUT_DIRECTORY)
    if(NOT BFPMX_LIB_PATH)
        set(BFPMX_LIB_PATH "${CMAKE_BINARY_DIR}/artifacts/lib")
    endif()
    
    # Get CUDA paths
    if(CUDAToolkit_FOUND)
        set(CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
        set(CUDA_LIB_DIR ${CUDAToolkit_LIBRARY_DIR})
    else()
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            set(CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
            set(CUDA_LIB_DIR ${CUDAToolkit_LIBRARY_DIR})
        else()
            # Fallback: try to find CUDA
            get_filename_component(CUDA_ROOT "${CMAKE_CUDA_COMPILER}" DIRECTORY)
            get_filename_component(CUDA_ROOT "${CUDA_ROOT}" DIRECTORY)
            set(CUDA_INCLUDE_DIRS "${CUDA_ROOT}/include")
            set(CUDA_LIB_DIR "${CUDA_ROOT}/lib64")
        endif()
    endif()
    
    # Set up source and output paths
    set(STDPAR_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/jacobi_stdpar.cpp")
    set(STDPAR_BINARY "${CMAKE_CURRENT_BINARY_DIR}/jacobi2d_stdpar")
    set(STDPAR_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/jacobi_stdpar.o")
    
    # Get include directories from bfpmx target
    get_target_property(BFPMX_INCLUDE_DIRS bfpmx INTERFACE_INCLUDE_DIRECTORIES)
    if(NOT BFPMX_INCLUDE_DIRS)
        set(BFPMX_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/mx")
    endif()
    
    # OpenMP is linked into bfpmx static library, but we may need runtime
    # nvc++ supports -mp (OpenMP) flag, but since bfpmx already has OpenMP,
    # we might not need it. However, let's add it to be safe.
    # Note: nvc++ uses -mp for OpenMP (not -fopenmp like GCC)
    set(OPENMP_LINK_FLAGS "-mp")
    
    # Custom compile command for stdpar with nvc++
    add_custom_command(
        OUTPUT ${STDPAR_BINARY}
        COMMAND ${NVCXX_COMPILER}
            -stdpar                    # Enable GPU offloading
            -std=c++20                 # C++20 standard
            -O3                        # Optimization
            -I${CMAKE_SOURCE_DIR}/mx   # Main include directory
            -I${CMAKE_CURRENT_SOURCE_DIR}  # Benchmarks directory (for jacobi_stdpar/)
            -I${CUDA_INCLUDE_DIRS}     # CUDA includes
            -DHAS_CUDA                 # Define CUDA flag
            -c ${STDPAR_SOURCE}
            -o ${STDPAR_OBJECT}
        COMMAND ${NVCXX_COMPILER}
            -stdpar                    # Also needed for linking
            ${STDPAR_OBJECT}
            -L${BFPMX_LIB_PATH}        # Link against bfpmx
            -lbfpmx
            -L${CUDA_LIB_DIR}          # Link CUDA runtime
            -lcudart
            ${OPENMP_LINK_FLAGS}       # Link OpenMP (bfpmx dependency)
            -o ${STDPAR_BINARY}
        DEPENDS ${STDPAR_SOURCE} bfpmx
        COMMENT "Building jacobi2d_stdpar with nvc++ and -stdpar flag"
        VERBATIM
    )
    
    add_custom_target(jacobi2d_stdpar ALL DEPENDS ${STDPAR_BINARY})
    
    # Make sure bfpmx is built first
    add_dependencies(jacobi2d_stdpar bfpmx)
    
    message(STATUS "jacobi2d_stdpar: Will be built with nvc++ at ${NVCXX_COMPILER}")
    message(STATUS "jacobi2d_stdpar: Using -stdpar flag for GPU offloading")
else()
    if(NOT NVCXX_COMPILER)
        message(STATUS "jacobi2d_stdpar: nvc++ not found, skipping stdpar target")
        message(STATUS "jacobi2d_stdpar: Install NVIDIA HPC SDK to enable GPU offloading")
    else()
        message(STATUS "jacobi2d_stdpar: CUDA not found, skipping stdpar target")
    endif()
endif()


add_executable(drift error/drift.cpp)
target_link_libraries(drift PRIVATE bfpmx)

add_executable(pure error/pure.cpp)
target_link_libraries(pure PRIVATE bfpmx)

# GPU Benchmarks #
if(CMAKE_CUDA_COMPILER)
    # Generate JacobiKernelInstantiations.cu from template headers
    set(JACOBI_INSTANTIATIONS_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/jacobi_gpu/generate_jacobi_instantiations.py")
    set(JACOBI_INSTANTIATIONS_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/jacobi_gpu/JacobiKernelInstantiations.cu")
    
    # Template header files that the script depends on
    set(JACOBI_TEMPLATE_HEADERS
        "${CMAKE_CURRENT_SOURCE_DIR}/jacobi_gpu/JacobiKernel.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/jacobi_gpu/JacobiKernelMxVectorNaive.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/jacobi_gpu/JacobiKernelMxVectorBlockWise.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/jacobi_gpu/JacobiKernelMxVectorBlockWiseRequant.cuh"
    )
    
    # Find Python3
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(NOT Python3_EXECUTABLE)
        find_program(PYTHON_EXECUTABLE python3 python)
        if(PYTHON_EXECUTABLE)
            set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
        endif()
    endif()
    
    # Custom command to generate instantiations
    # The script writes to the source directory, which is fine for version control
    add_custom_command(
        OUTPUT ${JACOBI_INSTANTIATIONS_SOURCE}
        COMMAND ${Python3_EXECUTABLE} "${JACOBI_INSTANTIATIONS_SCRIPT}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/jacobi_gpu"
        DEPENDS 
            ${JACOBI_INSTANTIATIONS_SCRIPT}
            ${JACOBI_TEMPLATE_HEADERS}
        COMMENT "Generating JacobiKernelInstantiations.cu from template headers"
        VERBATIM
    )
    
    add_executable(jacobi2d_gpu 
        jacobi_gpu/jacobi_gpu.cpp
        jacobi_gpu/JacobiKernel.cu
        ${JACOBI_INSTANTIATIONS_SOURCE}
        jacobi_gpu/PackKernelRequant.cu
    )
    target_include_directories(jacobi2d_gpu PRIVATE
        ${CMAKE_SOURCE_DIR}/mx
        ${CMAKE_SOURCE_DIR}/mx/arch/gpu
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/jacobi_gpu
    )
    target_link_libraries(jacobi2d_gpu 
        PRIVATE 
        bfpmx 
        bfpmx_gpu
        nvtx3-c
    )
    set_target_properties(jacobi2d_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()