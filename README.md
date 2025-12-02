## Quick Start

In order to get started, you'll need CMake itself.
You'll also need to ensure you have CMake v4 or above,
preferably CMake v4.1.2. You'll also need python of
a version above or equal to v3.9.

We use clang tools version 21. As such, `clang-tidy-21`, 
`clang-format-21` and `clang-21` are the valid binaries
for this program. Newer versions may work, however CI
will run formatting against v21, so there may be
discrepancies with differing versions.

Make sure you have `clang`, `clang-tidy` and `clang++` on
your path.

### MacOS Specifics

If you're on MacOS, you should also ensure you have
the Developer Tools extended features. You can do so,
if you haven't already, with the following command.

```bash
xcode-select --install
```
### Set up conda (recommended)
```
conda create -n bfpmx
conda activate bfpmx
conda install -c conda-forge cmake=4.2.0 cuda=12.9
```

### Linting

The following commands can be used to build the
CMake project binaries, and run the linter.

> The recommended IDE is JetBrains CLion, or VSCode
which will autodetect the config and inline linting
into the code itself.

```bash
# Create build plan for release type (i.e. Debug)
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# Create the makefile, for the build system
cmake .

# Build project binaries and library linkable
cmake --build build

# Run tidy on files
#
# Note: MacOS users may need to substitute for the following, 
#       after installing `llvm` through homebrew...
#       /opt/homebrew/opt/llvm/bin/run-clang-tidy
#
run-clang-tidy -p build mx/ -quiet
```