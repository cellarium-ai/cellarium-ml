#!/usr/bin/env bash
# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause
#
# Builds harness.cpp against the C++ sources of BPCells (MIT / Apache-2.0, https://github.com/bnprks/BPCells), downloaded at a pinned
# commit into ./build/. Nothing from BPCells is vendored in this repository. Needs: curl, tar and a C++17 compiler (CXX, default c++).
# The harness decodes with the vendored Highway SIMD library that ships inside the BPCells tarball.
set -euo pipefail
cd "$(dirname "$0")"

BPCELLS_COMMIT="016296413cd7a5725c8b35816df02365d7def853"
SRC="build/bpcells-${BPCELLS_COMMIT}"
OUT="build/bpcells_decode_harness"

if [ ! -d "$SRC" ]; then
  mkdir -p build
  echo "downloading BPCells ${BPCELLS_COMMIT} ..."
  curl -sSL --fail "https://github.com/bnprks/BPCells/archive/${BPCELLS_COMMIT}.tar.gz" -o build/bpcells.tar.gz
  mkdir -p "$SRC"
  tar xzf build/bpcells.tar.gz -C "$SRC" --strip-components=1
fi

CPP="$SRC/r/src/bpcells-cpp"
VENDOR="$SRC/r/src/vendor"
${CXX:-c++} -std=c++17 -O2 -pthread -Wno-unused-but-set-variable \
  -I "$CPP" -I "$VENDOR" -I "$VENDOR/highway" \
  harness.cpp \
  "$CPP/arrayIO/array_interfaces.cpp" "$CPP/arrayIO/bp128.cpp" "$CPP/arrayIO/vector.cpp" \
  "$CPP"/simd/bp128/*.cpp "$CPP/simd/current_target.cpp" \
  "$VENDOR/highway/hwy/targets.cc" "$VENDOR/highway/hwy/per_target.cc" "$VENDOR/highway/hwy/print.cc" \
  "$VENDOR/highway/hwy/timer.cc" "$VENDOR/highway/hwy/aligned_allocator.cc" \
  -o "$OUT"
echo "built $(pwd)/$OUT"
