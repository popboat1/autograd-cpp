# tensor engine unit tests
add_executable(tensorgrad_tests tensorgrad_tests.cpp)
add_test(NAME TensorGradMatrixTest COMMAND tensorgrad_tests)
target_link_libraries(tensorgrad_tests PRIVATE autograd_core)

# GPU tensor engine unit tests
add_executable(tensorgrad_gpu_tests tensorgrad_gpu_tests.cpp)
add_test(NAME TensorGradGPUTest COMMAND tensorgrad_gpu_tests)
target_link_libraries(tensorgrad_gpu_tests PRIVATE autograd_core)

# neural nets & optims tests
add_executable(nn_tests nn_tests.cpp)
add_test(NAME NNIntegrationTest COMMAND nn_tests)
target_link_libraries(nn_tests PRIVATE autograd_core)