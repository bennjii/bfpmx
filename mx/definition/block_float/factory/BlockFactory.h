//
// Created by Benjamin White on 10/10/2025.
//

#ifndef BFPMX_BLOCKFACTORY_H
#define BFPMX_BLOCKFACTORY_H

#include "definition/quantization/Quantize.h"

constexpr u16 DEFAULT_BLOCK_SIZE = 32;
constexpr u16 DEFAULT_BITS_SCALAR = 32;
constexpr u16 BITS_IN_BYTE = 8;

template <
    BlockDimsType BlockShape, typename Scalar, IFloatRepr Float,
    template <typename> typename ArithmeticPolicy,
    template <std::size_t, BlockDimsType, IFloatRepr> typename QuantizePolicy>
  requires IQuantize<QuantizePolicy, Scalar, BlockShape, Float,
                     ArithmeticPolicy>
class BlockFactory {
public:
  BlockFactory() = delete;

  /// The size of the block, in bits, when constructed.
  static constexpr u32 Size() {
    return Float::Size() * BlockShape::TotalSize();
  }

  static constexpr Block<Scalar, BlockShape, Float, ArithmeticPolicy,
                         QuantizePolicy>
  CreateBlock() {
    return Block<Scalar, BlockShape, Float, ArithmeticPolicy, QuantizePolicy>();
  }
};

#endif // BFPMX_BLOCKFACTORY_H