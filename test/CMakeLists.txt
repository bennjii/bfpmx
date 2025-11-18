add_executable(test_quantization test_quantization.cpp)
target_link_libraries(test_quantization PRIVATE bfpmx)

add_executable(test_large test_large.cpp)
target_link_libraries(test_large PRIVATE bfpmx)

add_test(NAME test_quantization COMMAND test_quantization)
add_test(NAME test_large COMMAND test_large)