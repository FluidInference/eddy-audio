if(EDDY_ENABLE_WHISPER)
  add_executable(whisper_example whisper_example.cpp)
  target_link_libraries(whisper_example PRIVATE eddy)
  install(TARGETS whisper_example DESTINATION bin)
endif()

add_executable(parakeet_cli parakeet_cli.cpp)
target_link_libraries(parakeet_cli PRIVATE eddy)

add_executable(hf_fetch_models hf_fetch_models.cpp)
# Include eddy headers for model_configs.hpp
target_include_directories(hf_fetch_models PRIVATE ${CMAKE_SOURCE_DIR}/include)

install(TARGETS parakeet_cli hf_fetch_models DESTINATION bin)
