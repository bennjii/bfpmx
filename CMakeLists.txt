cmake_minimum_required(VERSION 4.0)
project(bfpmx)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB_RECURSE SOURCES "mx/*.h")
file(GLOB_RECURSE SOURCES "mx/*.cpp")
add_library(bfpmx STATIC ${SOURCES})

if(APPLE)
    execute_process(
            COMMAND clang++ -v -x c++ -E /dev/null
            ERROR_VARIABLE CMAKE_CLANG_INCLUDE_PATHS
    )
    string(REGEX MATCHALL "-I[^ ]+" CMAKE_CLANG_INCLUDE_PATHS "${CMAKE_CLANG_INCLUDE_PATHS}")
    foreach(path ${CMAKE_CLANG_INCLUDE_PATHS})
        string(APPEND CMAKE_CXX_FLAGS " ${path}")
    endforeach()
endif()

target_include_directories(bfpmx PUBLIC mx)

add_definitions(-DUNIT_TESTS)

# TESTING #
add_executable(test_main test/main.cpp)
target_link_libraries(test_main PRIVATE bfpmx)

add_executable(test_marshal test/marshalling.cpp)
target_link_libraries(test_marshal PRIVATE bfpmx)

add_executable(test_playground test/playground.cpp)
target_link_libraries(test_playground PRIVATE bfpmx)