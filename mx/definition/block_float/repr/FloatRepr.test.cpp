//
// Created by Benjamin White on 10/10/2025.
//

#include "FloatRepr.h"

#include <catch2/catch_test_macros.hpp>

#include "definition/prelude.h"

TEST_CASE("FloatRepr Total Size") {
  SECTION("fp8") {
    REQUIRE(fp8::E4M3Type::Size() == 8);
    REQUIRE(fp8::E5M2Type::Size() == 8);
  }

  SECTION("fp6") {
    REQUIRE(fp6::E2M3Type::Size() == 6);
    REQUIRE(fp6::E3M2Type::Size() == 6);
  }

  SECTION("fp4") { REQUIRE(fp4::E2M1Type::Size() == 4); }
}

TEST_CASE("FloatRepr Nomenclature") {
  SECTION("fp8") {
    REQUIRE(fp8::E4M3Type::Nomenclature() == "FP8 E4M3");
    REQUIRE(fp8::E5M2Type::Nomenclature() == "FP8 E5M2");
  }

  SECTION("fp6") {
    REQUIRE(fp6::E2M3Type::Nomenclature() == "FP6 E2M3");
    REQUIRE(fp6::E3M2Type::Nomenclature() == "FP6 E3M2");
  }

  SECTION("fp4") { REQUIRE(fp4::E2M1Type::Nomenclature() == "FP4 E2M1"); }
}

TEST_CASE("FloatRepr Loss Limiter") {
  // Values selected are at the limit of what is representable by the
  // given datatype. This test ensures no regressions occur even within
  // some small range.

  SECTION("fp8") {
    // Exact Representation

    REQUIRE(fp8::E4M3Type::Loss(1.0) <= 0.0);
    REQUIRE(fp8::E5M2Type::Loss(1.0) <= 0.0);
    // Exact Representation

    REQUIRE(fp8::E4M3Type::Loss(5.0) <= 0.0);
    REQUIRE(fp8::E5M2Type::Loss(5.0) <= 0.0);

    // Exact Representation
    REQUIRE(fp8::E4M3Type::Loss(20.0) <= 0.0);
    REQUIRE(fp8::E5M2Type::Loss(20.0) <= 0.0);

    // Some loss within a given acceptance criteria
    REQUIRE(fp8::E4M3Type::Loss(200.0) <= 8.0);
    REQUIRE(fp8::E5M2Type::Loss(200.0) <= 8.0);
  }

  SECTION("fp6") {
    REQUIRE(fp6::E2M3Type::Loss(1.0) <= 1.0);
    REQUIRE(fp6::E3M2Type::Loss(1.0) <= 1.0);
  }

  SECTION("fp4") { REQUIRE(fp4::E2M1Type::Loss(1.0) <= 1.0); }
}