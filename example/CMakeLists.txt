# Examples #
add_executable(example_main main.cpp)
target_link_libraries(example_main PRIVATE bfpmx)

add_executable(example_marshal marshalling.cpp)
target_link_libraries(example_marshal PRIVATE bfpmx)

add_executable(example_marshal_profiler marshalling_with_profiler.cpp)
target_link_libraries(example_marshal_profiler PRIVATE bfpmx)

add_executable(example_playground playground.cpp)
target_link_libraries(example_playground PRIVATE bfpmx)

add_executable(example_quant quant.cpp)
target_link_libraries(example_quant PRIVATE bfpmx)

add_executable(example_mx_vector mx_vector.cpp)
target_link_libraries(example_mx_vector PRIVATE bfpmx)