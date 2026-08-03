# tensor engine unit tests
add_executable(tensorgrad_tests
    tensorgrad_tests.cpp
    ${CMAKE_SOURCE_DIR}/src/autograd/Tensor.cu
    ${CMAKE_SOURCE_DIR}/src/nn/Loss.cpp
)
add_test(NAME TensorGradMatrixTest COMMAND tensorgrad_tests)
target_link_libraries(tensorgrad_tests PRIVATE OpenMP::OpenMP_CXX CUDA::cublas CUDA::cudart)

# GPU tensor engine unit tests
add_executable(tensorgrad_gpu_tests
    tensorgrad_gpu_tests.cpp
    ${CMAKE_SOURCE_DIR}/src/autograd/Tensor.cu
)
add_test(NAME TensorGradGPUTest COMMAND tensorgrad_gpu_tests)
target_link_libraries(tensorgrad_gpu_tests PRIVATE OpenMP::OpenMP_CXX CUDA::cublas CUDA::cudart)

# neural nets & optims tests
add_executable(nn_tests
    nn_tests.cpp
    ${CMAKE_SOURCE_DIR}/src/autograd/Tensor.cu
    ${CMAKE_SOURCE_DIR}/src/nn/Linear.cpp
    ${CMAKE_SOURCE_DIR}/src/nn/MLP.cpp
    ${CMAKE_SOURCE_DIR}/src/nn/Loss.cpp
    ${CMAKE_SOURCE_DIR}/src/optim/SGD.cpp
    ${CMAKE_SOURCE_DIR}/src/optim/Adam.cpp
    ${CMAKE_SOURCE_DIR}/src/nn/Conv2D.cpp
    ${CMAKE_SOURCE_DIR}/src/nn/MaxPool2D.cpp
    ${CMAKE_SOURCE_DIR}/src/nn/AvgPool2D.cpp
    ${CMAKE_SOURCE_DIR}/src/nn/BatchNorm2D.cpp
    ${CMAKE_SOURCE_DIR}/src/nn/Sequential.cpp
)
add_test(NAME NNIntegrationTest COMMAND nn_tests)
target_link_libraries(nn_tests PRIVATE OpenMP::OpenMP_CXX CUDA::cublas CUDA::cudart)