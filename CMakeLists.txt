cmake_minimum_required(VERSION 4.0)
project(bfpmx)

set(CMAKE_CXX_STANDARD 23)

file(GLOB_RECURSE SOURCES "mx/*.h")
file(GLOB_RECURSE SOURCES "mx/*.cpp")
add_library(bfpmx STATIC ${SOURCES})

target_include_directories(bfpmx PUBLIC mx)

add_definitions(-DUNIT_TESTS)

# TESTING #
add_executable(test_main test/main.cpp)
target_link_libraries(test_main PRIVATE bfpmx)