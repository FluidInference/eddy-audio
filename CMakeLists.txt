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

option(EDDY_ENABLE_WHISPER "Enable Whisper (OpenVINO GenAI) support" OFF)
option(EDDY_BUILD_EXAMPLES "Build examples" ON)
option(EDDY_BUILD_CSHARP "Build C# bindings" OFF)

# Core dependencies
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(SndFile REQUIRED)
find_package(SampleRate REQUIRED)

# Whisper support is optional and requires OpenVINO GenAI
if(EDDY_ENABLE_WHISPER)
  find_package(OpenVINOGenAI REQUIRED)
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
    # Core utilities
    src/core/runtime.cpp
    src/core/model.cpp
    src/core/app_dir.cpp
    src/utils/audio_utils.cpp
    src/utils/ensure_models.cpp
    src/utils/tokenizer.cpp

    # OpenVINO backend
    src/backends/openvino_backend.cpp

    # Parakeet TDT v2 implementation
    src/models/parakeet-v2/parakeet_openvino.cpp
    src/models/parakeet-v2/parakeet_preprocessor.cpp
    src/models/parakeet-v2/parakeet_encoder.cpp
    src/models/parakeet-v2/parakeet_decoder.cpp
    src/models/parakeet-v2/parakeet_chunking.cpp
)

# Link dependencies
target_compile_definitions(eddy PUBLIC EDDY_WITH_OPENVINO=1)
target_link_libraries(eddy
  PUBLIC
    openvino::runtime
    SndFile::sndfile
    SampleRate::samplerate
)

if(WIN32)
  target_link_libraries(eddy PRIVATE Shell32)
endif()

# Optional Whisper support
if(EDDY_ENABLE_WHISPER)
  target_sources(eddy PRIVATE src/pipelines/whisper_pipeline.cpp)
  target_compile_definitions(eddy PUBLIC EDDY_WITH_OPENVINO_GENAI=1)
  target_link_libraries(eddy PUBLIC openvino::genai)
endif()

install(TARGETS eddy EXPORT eddyTargets ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(EXPORT eddyTargets FILE eddyTargets.cmake NAMESPACE eddy:: DESTINATION lib/cmake/eddy)
install(DIRECTORY include/ DESTINATION include)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/eddyConfigVersion.cmake
  VERSION 0.1.0
  COMPATIBILITY AnyNewerVersion
)
configure_file(cmake/eddyConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/eddyConfig.cmake @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/eddyConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/eddyConfigVersion.cmake DESTINATION lib/cmake/eddy)

if(EDDY_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# Build C API shim only when Whisper (GenAI) is available to satisfy symbols
# C API shim (always build; Whisper calls are no-ops unless EDDY_ENABLE_WHISPER)
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
