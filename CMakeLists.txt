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

file(GLOB_RECURSE SOURCES "mx/*.cpp")
list(APPEND SOURCES ${HEADERS})

add_library(bfpmx STATIC ${SOURCES})
target_include_directories(bfpmx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/mx)

set_target_properties(bfpmx PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/artifacts/lib"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/artifacts/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/artifacts/bin"
)

enable_testing()
add_subdirectory(test)
add_definitions(-DUNIT_TESTS)

# Example #
add_executable(example_main example/main.cpp)
target_link_libraries(example_main PRIVATE bfpmx)

add_executable(example_marshal example/marshalling.cpp)
target_link_libraries(example_marshal PRIVATE bfpmx)

add_executable(example_playground example/playground.cpp)
target_link_libraries(example_playground PRIVATE bfpmx)

add_executable(example_quant example/quant.cpp)
target_link_libraries(example_quant PRIVATE bfpmx)