add_executable(openvino_utils_test openvino_utils_test.cpp)
target_include_directories(openvino_utils_test PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(openvino_utils_test PRIVATE eddy)

add_test(NAME openvino_utils_test COMMAND openvino_utils_test)

add_executable(eddy_c_test eddy_c_test.cpp)
target_link_libraries(eddy_c_test PRIVATE eddy_c)
target_include_directories(eddy_c_test PRIVATE "${PROJECT_SOURCE_DIR}/include")

add_test(NAME eddy_c_test COMMAND eddy_c_test)
