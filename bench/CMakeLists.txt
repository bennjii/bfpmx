add_executable(bench_vector_ops mx_vec_ops.cpp)
target_link_libraries(bench_vector_ops PRIVATE bfpmx ${BENCHMARK_LIBS})