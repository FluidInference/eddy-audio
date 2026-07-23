add_executable(openvino_utils_test openvino_utils_test.cpp)
target_include_directories(openvino_utils_test PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(openvino_utils_test PRIVATE eddy)

add_test(NAME openvino_utils_test COMMAND openvino_utils_test)
