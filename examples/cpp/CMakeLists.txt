if(EDDY_ENABLE_WHISPER)
  add_executable(whisper_example whisper_example.cpp)
  target_link_libraries(whisper_example PRIVATE eddy)
  install(TARGETS whisper_example DESTINATION bin)
endif()

add_executable(parakeet_cli parakeet_cli.cpp)
target_link_libraries(parakeet_cli PRIVATE eddy)

add_executable(nemotron_cli nemotron_cli.cpp)
target_link_libraries(nemotron_cli PRIVATE eddy)

add_executable(hf_fetch_models hf_fetch_models.cpp)
target_link_libraries(hf_fetch_models PRIVATE eddy)

add_executable(benchmark_fleurs benchmark_fleurs.cpp)
target_link_libraries(benchmark_fleurs PRIVATE eddy)

add_executable(benchmark_nemotron_fleurs benchmark_nemotron_fleurs.cpp)
target_link_libraries(benchmark_nemotron_fleurs PRIVATE eddy)

install(TARGETS parakeet_cli nemotron_cli hf_fetch_models benchmark_fleurs benchmark_nemotron_fleurs DESTINATION bin)
