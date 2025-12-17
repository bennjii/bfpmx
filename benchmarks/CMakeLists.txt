
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

# GPU Benchmarks #
if(CMAKE_CUDA_COMPILER)
    add_executable(jacobi2d_gpu jacobi_gpu.cpp)
    target_link_libraries(jacobi2d_gpu 
        PRIVATE 
        bfpmx 
        bfpmx_gpu
    )
    set_target_properties(jacobi2d_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()