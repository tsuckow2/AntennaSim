cmake_minimum_required(VERSION 3.14)
project(nec2c-wasm C CXX)

set(MFEM_VERSION "v4.8" CACHE STRING "MFEM version or Git tag used by CPM")
set(MFEM_SOURCE_DIR "" CACHE PATH "Optional local MFEM source tree override")

# Bootstrap CPM in the build tree so MFEM is fetched and cached as a normal
# CMake dependency instead of being cloned by the shell build script.
set(CPM_VERSION "0.40.2")
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_VERSION}.cmake")
if(NOT EXISTS "${CPM_DOWNLOAD_LOCATION}")
  file(DOWNLOAD
    "https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_VERSION}/CPM.cmake"
    "${CPM_DOWNLOAD_LOCATION}"
    TLS_VERIFY ON
    STATUS CPM_DOWNLOAD_STATUS
  )
  list(GET CPM_DOWNLOAD_STATUS 0 CPM_DOWNLOAD_CODE)
  if(NOT CPM_DOWNLOAD_CODE EQUAL 0)
    message(FATAL_ERROR "Unable to download CPM.cmake: ${CPM_DOWNLOAD_STATUS}")
  endif()
endif()
include("${CPM_DOWNLOAD_LOCATION}")

# nec2c source files (flat layout — all .c files in submodule root)
set(NEC2C_DIR ${CMAKE_CURRENT_SOURCE_DIR}/nec2c)

set(NEC2C_SOURCES
  ${NEC2C_DIR}/main.c
  # nec2c.c is the OLD monolithic file — do NOT include it.
  # The refactored code lives in the separate .c files below.
  ${NEC2C_DIR}/calculations.c
  ${NEC2C_DIR}/fields.c
  ${NEC2C_DIR}/geometry.c
  ${NEC2C_DIR}/ground.c
  ${NEC2C_DIR}/input.c
  ${NEC2C_DIR}/matrix.c
  ${NEC2C_DIR}/misc.c
  ${NEC2C_DIR}/network.c
  ${NEC2C_DIR}/radiation.c
  ${NEC2C_DIR}/shared.c
  ${NEC2C_DIR}/somnec.c
)

add_executable(nec2c ${NEC2C_SOURCES})

target_include_directories(nec2c PRIVATE ${NEC2C_DIR})
target_link_libraries(nec2c m)

# Define PACKAGE_STRING (normally set by autoconf's config.h)
target_compile_definitions(nec2c PRIVATE
  PACKAGE_STRING="nec2c"
)

# Emscripten-specific flags
set_target_properties(nec2c PROPERTIES
  SUFFIX ".js"
)

target_link_options(nec2c PRIVATE
  # Module system
  -sEXPORT_NAME=createNec2c
  -sMODULARIZE=1
  # Memory
  -sALLOW_MEMORY_GROWTH=1
  -sINITIAL_MEMORY=33554432
  # Exported functions — we call main() programmatically
  -sEXPORTED_FUNCTIONS=_main
  -sEXPORTED_RUNTIME_METHODS=FS,callMain
  # Environment support
  -sENVIRONMENT=web,worker
  # Don't exit the runtime after main() returns
  -sEXIT_RUNTIME=0
  # Optimization
  -O3
)

add_executable(fem1d fem1d.cpp)
set_target_properties(fem1d PROPERTIES SUFFIX ".js")
target_link_options(fem1d PRIVATE
  -sEXPORT_NAME=createFem1d
  -sMODULARIZE=1
  -sALLOW_MEMORY_GROWTH=1
  -sINITIAL_MEMORY=16777216
  -sEXPORTED_FUNCTIONS=_fem_solve,_malloc,_free
  -sEXPORTED_RUNTIME_METHODS=HEAPF64
  -sENVIRONMENT=web,worker
  -sEXIT_RUNTIME=0
  -O3
)

if(MFEM_SOURCE_DIR AND EXISTS "${MFEM_SOURCE_DIR}/CMakeLists.txt")
  CPMAddPackage(NAME mfem SOURCE_DIR "${MFEM_SOURCE_DIR}")
else()
  set(MFEM_USE_MPI OFF CACHE BOOL "" FORCE)
  set(MFEM_USE_OPENMP OFF CACHE BOOL "" FORCE)
  set(MFEM_USE_LAPACK OFF CACHE BOOL "" FORCE)
  set(MFEM_USE_HYPRE OFF CACHE BOOL "" FORCE)
  set(MFEM_USE_METIS OFF CACHE BOOL "" FORCE)
  set(MFEM_USE_SUITESPARSE OFF CACHE BOOL "" FORCE)
  set(MFEM_USE_GSLIB OFF CACHE BOOL "" FORCE)
  set(MFEM_USE_NETCDF OFF CACHE BOOL "" FORCE)
  set(MFEM_USE_CURL OFF CACHE BOOL "" FORCE)
  CPMAddPackage(
    NAME mfem
    GITHUB_REPOSITORY mfem/mfem
    GIT_TAG "${MFEM_VERSION}"
    OPTIONS
      "MFEM_USE_MPI OFF"
      "MFEM_USE_OPENMP OFF"
      "MFEM_USE_LAPACK OFF"
      "MFEM_USE_HYPRE OFF"
      "MFEM_USE_METIS OFF"
      "MFEM_USE_SUITESPARSE OFF"
      "MFEM_USE_GSLIB OFF"
      "MFEM_USE_NETCDF OFF"
      "MFEM_USE_CURL OFF"
  )
endif()

if(TARGET mfem)
  add_executable(mfem_smoke mfem_smoke.cpp)
  target_link_libraries(mfem_smoke PRIVATE mfem)
  set_target_properties(mfem_smoke PROPERTIES SUFFIX ".js")
  target_link_options(mfem_smoke PRIVATE
    -sEXPORT_NAME=createMfemSmoke
    -sMODULARIZE=1
    -sALLOW_MEMORY_GROWTH=1
    -sINITIAL_MEMORY=33554432
    -sEXPORTED_FUNCTIONS=_mfem_hcurl_smoke
    -sENVIRONMENT=web,worker
    -sEXIT_RUNTIME=0
    -O3
  )

  add_executable(mfem_solver mfem_solver.cpp)
  target_link_libraries(mfem_solver PRIVATE mfem)
  set_target_properties(mfem_solver PROPERTIES SUFFIX ".js")
  target_link_options(mfem_solver PRIVATE
    -sEXPORT_NAME=createMfemSolver
    -sMODULARIZE=1
    -sALLOW_MEMORY_GROWTH=1
    -sINITIAL_MEMORY=33554432
    -sEXPORTED_FUNCTIONS=_mfem_solver_alloc,_mfem_solver_free,_mfem_solver_set_cancelled,_mfem_solver_run
    -sEXPORTED_RUNTIME_METHODS=HEAPU8
    -sENVIRONMENT=web,worker
    -sEXIT_RUNTIME=0
    -O3
  )
else()
  message(FATAL_ERROR "MFEM dependency did not provide the mfem target")
endif()
