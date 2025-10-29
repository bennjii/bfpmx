//
// Created by Benjamin White on 10/10/2025.
//

#include <catch2/catch_test_macros.hpp>

#include "definition/prelude.h"
#include "arch/prelude.h"

using Factory = BlockFactory<BlockDims<2, 2, 2>, 16, fp8::E4M3Type, CPUArithmetic, MaximumFractionalQuantization>;

TEST_CASE( "Block Factory Size Correct", "[BlockFactory]" ) {
    SECTION( "sizing should match" ) {
        REQUIRE( Factory::Size() == 8 * 8 );
    }
}