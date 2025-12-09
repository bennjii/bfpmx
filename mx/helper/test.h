// Here we put test helper functions

#ifndef BFPMX_ARCH_TEST_H
#define BFPMX_ARCH_TEST_H

#include <random>

template <typename T, std::size_t N>
static std::array<T, N> fill_random_arrays(const T low, const T high, const int seed = 25) {
  static std::mt19937_64 rng(seed);

  std::array<T, N> v;
  std::uniform_real_distribution<T> dist(low, high);
  for (std::size_t i = 0; i < N; i++)
    v[i] = dist(rng);
  return v;
}

template <typename T, std::size_t N>
static std::array<T, N> fill_known_arrays(const T value) {
  std::array<T, N> v;
  for (std::size_t i = 0; i < N; i++)
    v[i] = value;
  return v;
}

#endif // BFPMX_ARCH_TEST_H
