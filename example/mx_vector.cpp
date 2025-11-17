// /Users/minh/Code/bfpmx/example/mx_vector.cpp
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "definition/vector/MxVector.hpp"
#include "definition/prelude.h"
#include "definition/vector/MxVectorOperations.hpp"

int main() {
    // seed RNG
    std::mt19937_64 rng(
        static_cast<unsigned long long>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()
        )
    );

    // pick a random size and fill the mx::vector with random doubles
    std::uniform_int_distribution<int> size_dist(1, 100);
    std::uniform_real_distribution<double> val_dist(-5.0, 5.0);

    size_t n = 64;
    std::vector<f64> v1, v2;
    v1.reserve(n);
    v2.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        v1.push_back(i);
        v2.push_back(n-i);
    }

    // print size
    std::cout << "std::vector size: " << sizeof(f64) * v1.size() << std::endl;
    mx::vector::MxVector<BlockDims<16>> mx_vec1(v1), mx_vec2(v2);
    std::cout << "mx::vector internal size: " << mx_vec1.SizeInBytes() << std::endl;
    mx_vec1.asString();

    auto dot_prod = mx::vector::ops::Dot(mx_vec1, mx_vec2);
    std::cout << "Dot v1*v2 = " << dot_prod << std::endl;
    
    std::cout << "Sum v1+v2 =";
    mx::vector::ops::Add(mx_vec1, mx_vec2).asString();
    return 0;
}