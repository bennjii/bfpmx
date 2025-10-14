#!/bin/bash

# Get the actual compiler being used
CXX="${CXX:-clang++}"

# Extract system include paths from the compiler
SYSTEM_FLAGS=$($CXX -v -x c++ /dev/null -fsyntax-only 2>&1 | \
  awk '/#include <...> search starts here:/{flag=1;next}/End of search list/{flag=0}flag' | \
  sed 's/^[ \t]*//' | \
  sed 's/^/-isystem/' | \
  tr '\n' ' ')

# Also get the sysroot if on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
  SYSROOT=$($CXX -print-sysroot 2>/dev/null)
  if [ -n "$SYSROOT" ]; then
    SYSTEM_FLAGS="$SYSTEM_FLAGS -isysroot $SYSROOT"
  fi
fi

# Run clang-tidy with the discovered system flags
clang-tidy "$@" -- $SYSTEM_FLAGS