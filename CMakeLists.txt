cmake_minimum_required(VERSION 3.31.6)
project(bfpmx)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_custom_target(fix_compile_commands ALL
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/fix-compile-commands.py
        ${CMAKE_BINARY_DIR}/compile_commands.json
        DEPENDS ${CMAKE_BINARY_DIR}/compile_commands.json
        COMMENT "Fixing compile_commands.json for clang-tidy"
)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "mx/*.cpp")
list(APPEND SOURCES ${HEADERS})

# Remove any matching *.test.cpp
foreach(f ${SOURCES})
    if (f MATCHES ".*\\.test\\.cpp$")
        list(REMOVE_ITEM SOURCES "${f}")
    endif()
endforeach()

add_library(bfpmx STATIC ${SOURCES})
target_include_directories(bfpmx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/mx)

set_target_properties(bfpmx PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/artifacts/lib"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/artifacts/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/artifacts/bin"
)

set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/.deps")
Include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.8.1 # or a later release
)

FetchContent_MakeAvailable(Catch2)

# Required step: https://github.com/catchorg/Catch2/blob/devel/docs/cmake-integration.md#top
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

include(CTest)
include(Catch)

enable_testing()

file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/*\.test.cpp"
)

foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)

    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE
            Catch2::Catch2WithMain
            bfpmx
    )

    catch_discover_tests(${test_name})
endforeach()

add_subdirectory(test)
add_definitions(-DUNIT_TESTS)

# CUDA
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA is OK")
    enable_language(CUDA)
    # Variable to check in C++ code
    add_definitions(-DHAS_CUDA)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CUDA_SEPARABLE_COMPILATION ON)
    # Experimental relaxed calling host constexpr in cuda device code
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    add_executable(example_cudatest example/cudatest.cu)
    target_link_libraries(example_cudatest PRIVATE bfpmx)
else()
    message(STATUS "No CUDA")
endif()

# Examples #
add_executable(example_main example/main.cpp)
target_link_libraries(example_main PRIVATE bfpmx)

add_executable(example_marshal example/marshalling.cpp)
target_link_libraries(example_marshal PRIVATE bfpmx)

add_executable(example_marshal_profiler example/marshalling_with_profiler.cpp)
target_link_libraries(example_marshal_profiler PRIVATE bfpmx)

add_executable(example_playground example/playground.cpp)
target_link_libraries(example_playground PRIVATE bfpmx)

add_executable(example_quant example/quant.cpp)
target_link_libraries(example_quant PRIVATE bfpmx)

# Benchmarks #
add_executable(jacobi2d benchmarks/jacobi.cpp)
target_link_libraries(jacobi2d PRIVATE bfpmx)

add_executable(heat3d benchmarks/heat3d.cpp)
target_link_libraries(heat3d PRIVATE bfpmx)

add_executable(error_trend benchmarks/error_trend.cpp)
target_link_libraries(error_trend PRIVATE bfpmx)