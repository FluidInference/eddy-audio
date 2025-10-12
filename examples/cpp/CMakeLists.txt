add_executable(whisper_example whisper_example.cpp)
target_link_libraries(whisper_example PRIVATE eddy)

add_executable(test_parakeet test_parakeet.cpp)
target_link_libraries(test_parakeet PRIVATE eddy)

add_executable(parakeet_cli parakeet_cli.cpp)
target_link_libraries(parakeet_cli PRIVATE eddy)

install(TARGETS whisper_example parakeet_cli DESTINATION bin)