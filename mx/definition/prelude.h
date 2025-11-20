//
// Created by Benjamin White on 10/10/2025.
//

#ifndef BFPMX_PRELUDE_H
#define BFPMX_PRELUDE_H

#include "alias.h"
#include "block_float/repr/FloatRepr.h"

#include "block_float/block/Block.h"
#include "block_float/factory/BlockFactory.h"

#include "quantization/prelude.h"

#include "util.cpp"
#ifdef HAS_CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda/std/array>
#endif

#endif // BFPMX_PRELUDE_H