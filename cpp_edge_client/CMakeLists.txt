cmake_minimum_required(VERSION 3.14)
project(SplitOculoEdgeClient)

set(CMAKE_CXX_STANDARD 17)

# OpenCV
find_package(OpenCV REQUIRED)

# ONNX Runtime
# Assuming ONNX Runtime is installed or provided via CMAKE_PREFIX_PATH
find_package(OnnxRuntime)
if (NOT OnnxRuntime_FOUND)
    # Simple search for header and lib if find_package fails
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        HINTS ${ONNXRUNTIME_ROOT}/include)
    find_library(ONNXRUNTIME_LIB onnxruntime
        HINTS ${ONNXRUNTIME_ROOT}/lib)
    if (ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
        message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIB}")
        include_directories(${ONNXRUNTIME_INCLUDE_DIR})
    else()
        message(FATAL_ERROR "ONNX Runtime not found. Please set ONNXRUNTIME_ROOT.")
    endif()
endif()

# httplib (header only)
# We will use FetchContent to get it automatically
include(FetchContent)
FetchContent_Declare(
    cpp-httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(cpp-httplib)

# JSON (nlohmann/json)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.2
)
FetchContent_MakeAvailable(json)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(edge_client src/main.cpp)

target_link_libraries(edge_client
    PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
    httplib::httplib
    nlohmann_json::nlohmann_json
)

if (WIN32)
    target_link_libraries(edge_client PRIVATE ws2_32 crypt32)
endif()
