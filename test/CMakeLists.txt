add_executable(test_math test_math.cpp)
target_link_libraries(test_math PRIVATE bfpmx)

add_test(NAME math_test COMMAND test_math)
