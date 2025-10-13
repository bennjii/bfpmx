cmake_minimum_required(VERSION 4.0)
project(bfpmx)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_custom_target(fix_compile_commands ALL
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/fix-compile-commands.py
        ${CMAKE_BINARY_DIR}/compile_commands.json
        DEPENDS ${CMAKE_BINARY_DIR}/compile_commands.json
        COMMENT "Fixing compile_commands.json for clang-tidy"
)

file(GLOB_RECURSE SOURCES "mx/*.h")
file(GLOB_RECURSE SOURCES "mx/*.cpp")
add_library(bfpmx STATIC ${SOURCES})

target_include_directories(bfpmx PUBLIC mx)

add_definitions(-DUNIT_TESTS)

# TESTING #
add_executable(test_main test/main.cpp)
target_link_libraries(test_main PRIVATE bfpmx)

add_executable(test_marshal test/marshalling.cpp)
target_link_libraries(test_marshal PRIVATE bfpmx)

add_executable(test_playground test/playground.cpp)
target_link_libraries(test_playground PRIVATE bfpmx)