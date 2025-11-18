// Here we put test helper functions

#ifndef BFPMX_ARCH_TEST_H
#define BFPMX_ARCH_TEST_H

#include <catch2/catch_test_macros.hpp>
#include <catch2/interfaces/catch_interfaces_config.hpp>
#include <catch2/internal/catch_context.hpp>
#include <random>

template <typename T, std::size_t N>
std::array<T, N> fill_random_arrays(T low, T high) {
  static std::mt19937_64 rng(Catch::getCurrentContext().getConfig()->rngSeed());

  std::array<T, N> v;
  std::uniform_real_distribution<T> dist(low, high);
  for (std::size_t i = 0; i < N; i++)
    v[i] = dist(rng);
  return v;
}

#endif // BFPMX_ARCH_TEST_H
