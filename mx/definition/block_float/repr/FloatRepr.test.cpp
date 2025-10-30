//
// Created by Benjamin White on 10/10/2025.
//

#include "FloatRepr.h"

#include <catch2/catch_test_macros.hpp>

#include "definition/prelude.h"

TEST_CASE( "FloatRepr Total Size" ) {
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

TEST_CASE( "FloatRepr Nomenclature" ) {
  SECTION( "fp8" ) {
    REQUIRE( fp8::E4M3Type::Nomenclature() == "FP8 E4M3" );
    REQUIRE( fp8::E5M2Type::Nomenclature() == "FP8 E5M2" );
  }

  SECTION( "fp6" ) {
    REQUIRE( fp6::E2M3Type::Nomenclature() == "FP6 E2M3" );
    REQUIRE( fp6::E3M2Type::Nomenclature() == "FP6 E3M2" );
  }

  SECTION( "fp4" ) {
    REQUIRE( fp4::E2M1Type::Nomenclature() == "FP4 E2M1" );
  }
}

TEST_CASE( "FloatRepr Loss Limiter" ) {
  SECTION( "fp8" ) {
    REQUIRE( fp8::E4M3Type::Loss(1.0) <= 0.125 );
    REQUIRE( fp8::E5M2Type::Loss(1.0) <= 0.125 );
  }

  SECTION( "fp6" ) {
    REQUIRE( fp6::E2M3Type::Loss(1.0) <= 1.0 );
    REQUIRE( fp6::E3M2Type::Loss(1.0) <= 1.0 );
  }

  SECTION( "fp4" ) {
    REQUIRE( fp4::E2M1Type::Loss(1.0) <= 1.0 );
  }
}