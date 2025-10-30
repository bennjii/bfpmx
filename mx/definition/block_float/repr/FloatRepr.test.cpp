//
// Created by Benjamin White on 10/10/2025.
//

#include "FloatRepr.h"

#include <catch2/catch_test_macros.hpp>

#include "definition/prelude.h"

TEST_CASE( "FloatRepr Total Size Correct", "[FloatRepr]" ) {
  SECTION( "fp8" ) {
    REQUIRE( fp8::E4M3Type::Size() == 8 );
    REQUIRE( fp8::E5M2Type::Size() == 8 );
  }

  SECTION( "fp6" ) {
    REQUIRE( fp6::E2M3Type::Size() == 6 );
    REQUIRE( fp6::E3M2Type::Size() == 6 );
  }

  SECTION( "fp4" ) {
    REQUIRE( fp4::E2M1Type::Size() == 4 );
  }
}