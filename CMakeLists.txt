cmake_minimum_required(VERSION 3.22)
project(eddy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Enable interprocedural optimization (LTO/LTCG) for Release builds
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

if(MSVC)
  # Favor fast math and aggressive inlining for Release
  # Use flags string to avoid malformed generator expressions on MSBuild
  string(APPEND CMAKE_CXX_FLAGS_RELEASE " /O2 /Oi /Ot /Ob3 /fp:fast")
endif()

option(EDDY_ENABLE_OPENVINO "Enable OpenVINO backend" ON)
option(EDDY_BUILD_EXAMPLES "Build examples" ON)
option(EDDY_BUILD_CSHARP "Build C# bindings" OFF)

if(EDDY_ENABLE_OPENVINO)
  find_package(OpenVINO REQUIRED COMPONENTS Runtime)
  find_package(OpenVINOGenAI)  # Optional - only needed for Whisper
endif()

add_library(eddy STATIC)
target_include_directories(eddy
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_sources(eddy
  PRIVATE
    src/core/runtime.cpp
    src/core/model.cpp
    src/core/cache.cpp
    src/backends/openvino_backend.cpp
)

# Link Windows shell library for cache directory access
if(WIN32)
  target_link_libraries(eddy PRIVATE Shell32)
endif()

if(EDDY_ENABLE_OPENVINO)
  # Core OpenVINO-dependent sources (do not require GenAI)
  target_sources(eddy PRIVATE
    src/pipelines/audio_utils.cpp
    src/models/parakeet/parakeet_openvino.cpp
    src/streaming/buffered_streaming_asr.cpp
  )
  target_compile_definitions(eddy PUBLIC EDDY_WITH_OPENVINO=1)
  target_link_libraries(eddy PUBLIC openvino::runtime)

  # Link GenAI only if found (for Whisper support)
  if(OpenVINOGenAI_FOUND)
    # Whisper example depends on OpenVINO GenAI headers/libraries
    target_sources(eddy PRIVATE src/pipelines/whisper_pipeline.cpp)
    target_compile_definitions(eddy PUBLIC EDDY_WITH_OPENVINO_GENAI=1)
    target_link_libraries(eddy PUBLIC openvino::genai)
  endif()
endif()

install(TARGETS eddy EXPORT eddyTargets ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(EXPORT eddyTargets FILE eddyTargets.cmake NAMESPACE eddy:: DESTINATION lib/cmake/eddy)
install(DIRECTORY include/ DESTINATION include)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/eddyConfigVersion.cmake
  VERSION 0.1.0
  COMPATIBILITY AnyNewerVersion
)
set(EDDY_ENABLE_OPENVINO_CONFIG ${EDDY_ENABLE_OPENVINO})
configure_file(cmake/eddyConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/eddyConfig.cmake @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/eddyConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/eddyConfigVersion.cmake DESTINATION lib/cmake/eddy)

if(EDDY_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# Build a shared C API shim (eddy_c) that exposes the extern "C" FFI
add_library(eddy_c SHARED src/eddy_c.cpp)
target_include_directories(eddy_c
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(eddy_c PRIVATE eddy)
target_compile_definitions(eddy_c PRIVATE EDDY_BUILD_SHARED=1)

install(TARGETS eddy_c RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)

include(CTest)
# Only add tests if requested and the folder exists
if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  add_subdirectory(tests)
endif()
