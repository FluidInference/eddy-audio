if(OpenVINOGenAI_FOUND)
  add_executable(whisper_example whisper_example.cpp)
  target_link_libraries(whisper_example PRIVATE eddy)
  install(TARGETS whisper_example DESTINATION bin)
endif()

add_executable(parakeet_cli parakeet_cli.cpp)
target_link_libraries(parakeet_cli PRIVATE eddy)

add_executable(benchmark_librispeech benchmark_librispeech.cpp)
target_link_libraries(benchmark_librispeech PRIVATE eddy)
# Ensure nlohmann/json single-header is visible to example headers
target_include_directories(benchmark_librispeech PRIVATE ${CMAKE_SOURCE_DIR}/third_party)

# Optional: Hugging Face fetch helper (pure C++)
add_executable(hf_fetch_models hf_fetch_models.cpp)
# Try to use libcurl if available; otherwise the tool will fall back to invoking system 'curl'
find_package(CURL QUIET)
if(CURL_FOUND)
  target_link_libraries(hf_fetch_models PRIVATE CURL::libcurl)
  target_compile_definitions(hf_fetch_models PRIVATE HF_FETCH_HAVE_LIBCURL=1)
endif()

install(TARGETS parakeet_cli benchmark_librispeech hf_fetch_models DESTINATION bin)
