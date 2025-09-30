add_executable(whisper_example whisper_example.cpp)
target_link_libraries(whisper_example PRIVATE eddy)

install(TARGETS whisper_example DESTINATION bin)