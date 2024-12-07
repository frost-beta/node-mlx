# Required by cmake-js.
# https://github.com/cmake-js/cmake-js#node-api-and-node-addon-api
cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0091 NEW)
cmake_policy(SET CMP0042 NEW)

# Define project name.
project(node-mlx)

# Use C++17.
set(CMAKE_CXX_STANDARD 17)

# Define NAPI_VERSION.
add_definitions(-DNAPI_VERSION=9)

# Use cmake-js.
add_library(${PROJECT_NAME} SHARED ${CMAKE_JS_SRC})
target_include_directories(${PROJECT_NAME} PRIVATE "." ${CMAKE_JS_INC})
target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB})
set_target_properties(${PROJECT_NAME} PROPERTIES
                      PREFIX ""
                      OUTPUT_NAME "mlx"
                      SUFFIX ".node")
if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
  execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF} /out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
endif()

# Add source files.
file(GLOB SOURCE_FILES "src/*")
target_sources(${PROJECT_NAME} PRIVATE ${SOURCE_FILES})

# Use kizunapi.
target_include_directories(${PROJECT_NAME} PRIVATE "deps/kizunapi")

# Work around https://github.com/ml-explore/mlx/pull/1625.
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND ${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
  set(ACCELERATE_LIBRARY OFF)
endif()

# Link with MLX.
option(MLX_BUILD_TESTS "Build tests for mlx" OFF)
option(MLX_BUILD_EXAMPLES "Build examples for mlx" OFF)
option(MLX_ENABLE_X64_MAC "Enable building for x64 macOS" ON)
add_subdirectory(deps/mlx)
target_link_libraries(${PROJECT_NAME} mlx)
