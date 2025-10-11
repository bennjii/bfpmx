//
// Created by Benjamin White on 10/10/2025.
//

#ifndef BFPMX_BLOCK_H
#define BFPMX_BLOCK_H

#include <array>
#include <iostream>

#include "arch/prelude.h"
#include "definition/alias.h"

template<
    std::size_t BlockQuantity,
    std::size_t BlockSize,
    template<typename> typename ImplPolicy
>
class Block : public ArithmeticEnabled<
    Block<BlockQuantity, BlockSize, ImplPolicy>,
    ImplPolicy<Block<BlockQuantity, BlockSize, ImplPolicy>>
> {
public:
    static constexpr std::size_t TotalElements = BlockQuantity * BlockSize;
    using value_type = f64;

    Block() { data_.fill(0); }
    void Fill(f64 value)
    {
        data_.fill(value);
    }

    explicit Block(std::array<value_type, TotalElements> init) : data_(init) {}

    [[nodiscard]] static std::size_t Length()
    {
        return BlockQuantity;
    }

    value_type* data() { return data_.data(); }
    [[nodiscard]] const value_type* data() const { return data_.data(); }

    void print(const char* label = "") const {
        if (*label) std::cout << label << ": ";
        for (auto v : data_) std::cout << v << ' ';
        std::cout << '\n';
    }

private:
    std::array<value_type, TotalElements> data_;
};

#endif //BFPMX_BLOCK_H