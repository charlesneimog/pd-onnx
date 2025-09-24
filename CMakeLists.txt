cmake_minimum_required(VERSION 3.25)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(onnx)
endif()

# Global Configurations
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POLICY_VERSION_MINIMUM "3.5")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ╭──────────────────────────────────────╮
# │               pd.cmake               │
# ╰──────────────────────────────────────╯
set(PDCMAKE_FILE ${CMAKE_BINARY_DIR}/pd.cmake)
set(PDCMAKE_VERSION "v0.2.6")
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
option(BUILD_TESTS OFF)
option(BUILD_EXAMPLES OFF)
add_subdirectory(libonnx)

# ╭──────────────────────────────────────╮
# │            Pure Data external        │
# ╰──────────────────────────────────────╯
pd_add_external(pdonnx ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx.cpp EXTERNAL_NAME
                onnx LINK_LIBRARIES onnx)

pd_add_datafile(pdonnx "${CMAKE_CURRENT_SOURCE_DIR}/resources/onnx-help.pd")
pd_add_datafile(pdonnx "${CMAKE_CURRENT_SOURCE_DIR}/resources/train.pd")
pd_add_datafile(pdonnx "${CMAKE_CURRENT_SOURCE_DIR}/resources/model.onnx")
