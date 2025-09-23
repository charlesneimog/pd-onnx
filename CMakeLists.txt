cmake_minimum_required(VERSION 3.25)
project(onnx)

# Global Configurations
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POLICY_VERSION_MINIMUM "3.5")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ╭──────────────────────────────────────╮
# │               pd.cmake               │
# ╰──────────────────────────────────────╯
set(PDCMAKE_FILE ${CMAKE_BINARY_DIR}/pd.cmake)
set(PDCMAKE_VERSION "v0.2.5")
if(NOT EXISTS "${PDCMAKE_FILE}")
  file(
    DOWNLOAD
    https://raw.githubusercontent.com/pure-data/pd.cmake/refs/tags/${PDCMAKE_VERSION}/pd.cmake
    ${PDCMAKE_FILE})
endif()
include(${PDCMAKE_FILE})

# ╭──────────────────────────────────────╮
# │            ONNX Runtime              │
# ╰──────────────────────────────────────╯
# libonnx C library

option(BUILD_TESTS OFF)
option(BUILD_EXAMPLES OFF)

add_subdirectory(libonnx)
target_compile_options(onnx PRIVATE -ffunction-sections -fdata-sections)
target_link_options(onnx PRIVATE -Wl,--gc-sections)
target_compile_options(
  onnx
  PUBLIC -O3
         -march=native
         -mtune=native
         -ffast-math
         -funroll-loops
         -fstrict-aliasing
         -ffunction-sections
         -fdata-sections)

# ╭──────────────────────────────────────╮
# │            Pure Data external        │
# ╰──────────────────────────────────────╯
pd_add_external(pdonnx onnx.cpp EXTERNAL_NAME onnx LINK_LIBRARIES onnx)
