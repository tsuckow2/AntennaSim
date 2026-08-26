#!/bin/bash
# Build nec2c to WebAssembly using Emscripten
#
# Prerequisites:
#   - Emscripten SDK installed (the workspace-local /workspaces/emsdk is detected)
#   - nec2c submodule initialized (git submodule update --init)
#   - Network access on the first build so CPM can fetch MFEM v4.8
#
# Output:
#   build/nec2c.js    — JavaScript loader/glue code
#   build/nec2c.wasm  — WebAssembly binary
#   build/mfem_smoke.js/.wasm — MFEM H(curl) feasibility module
#
# Usage:
#   cd wasm && ./build.sh
#
# Or from project root:
#   ./wasm/build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load the workspace-local Emscripten SDK when it has not already been activated.
if ! command -v emcmake &> /dev/null && [ -f "/workspaces/emsdk/emsdk_env.sh" ]; then
    # shellcheck disable=SC1091
    source /workspaces/emsdk/emsdk_env.sh
fi

# Verify emscripten is available
if ! command -v emcmake &> /dev/null; then
    echo "ERROR: Emscripten not found. Install emsdk and run 'source emsdk_env.sh' first."
    echo "  See: https://emscripten.org/docs/getting_started/downloads.html"
    exit 1
fi

# Verify nec2c submodule is present
if [ ! -f "nec2c/nec2c.c" ]; then
    echo "ERROR: nec2c submodule not found. Run: git submodule update --init --recursive"
    exit 1
fi

# Apply patches if not already applied
if [ -f "patches/emscripten-compat.patch" ]; then
    echo "Checking if patches need to be applied..."
    cd nec2c
    if git apply --check ../patches/emscripten-compat.patch 2>/dev/null; then
        echo "Applying Emscripten compatibility patches..."
        git apply ../patches/emscripten-compat.patch
    else
        echo "Patches already applied or not applicable (skipping)."
    fi
    cd ..
fi

# Build
echo "Building nec2c to WebAssembly..."
mkdir -p build
cd build

CPM_SOURCE_CACHE="${CPM_SOURCE_CACHE:-$SCRIPT_DIR/build/cpm-cache}"
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DMFEM_VERSION="${MFEM_VERSION:-v4.8}"
    -DCPM_SOURCE_CACHE="$CPM_SOURCE_CACHE"
    -DMFEM_SOURCE_DIR="${MFEM_SOURCE_DIR:-}"
)
if [ -n "${MFEM_SOURCE_DIR:-}" ]; then
    CMAKE_ARGS[-1]="-DMFEM_SOURCE_DIR=$MFEM_SOURCE_DIR"
fi
emcmake cmake .. "${CMAKE_ARGS[@]}"
emmake make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Copy browser artifacts to the Vite public directory when building in-repo.
if [ -d "../../frontend/public/wasm" ]; then
    cp nec2c.js nec2c.wasm fem1d.js fem1d.wasm ../../frontend/public/wasm/
    if [ -f mfem_smoke.js ] && [ -f mfem_smoke.wasm ] && [ -f mfem_solver.js ] && [ -f mfem_solver.wasm ]; then
        cp mfem_smoke.js mfem_smoke.wasm mfem_solver.js mfem_solver.wasm ../../frontend/public/wasm/
    fi
fi

echo ""
echo "Build complete!"
echo "  NEC2C JS loader:  $(pwd)/nec2c.js"
echo "  FEM JS loader:    $(pwd)/fem1d.js"
echo "  WASM binaries:    $(pwd)/nec2c.wasm $(pwd)/fem1d.wasm"
echo "  MFEM smoke loader:  $(pwd)/mfem_smoke.js"
echo "  MFEM solver loader: $(pwd)/mfem_solver.js"
echo "  MFEM WASM binaries: $(pwd)/mfem_smoke.wasm $(pwd)/mfem_solver.wasm"
ls -lh nec2c.js nec2c.wasm fem1d.js fem1d.wasm mfem_smoke.js mfem_smoke.wasm mfem_solver.js mfem_solver.wasm 2>/dev/null || echo "(files not found — check build output above)"
