# tensor engine unit tests
add_executable(tensorgrad_tests
    tensorgrad_tests.cpp
    ${CMAKE_SOURCE_DIR}/src/autograd/Tensor.cpp
    ${CMAKE_SOURCE_DIR}/src/nn/Loss.cpp
)
add_test(NAME TensorGradMatrixTest COMMAND tensorgrad_tests)

# neural nets & optims tests
add_executable(nn_tests
    nn_tests.cpp
    ${CMAKE_SOURCE_DIR}/src/autograd/Tensor.cpp
    ${CMAKE_SOURCE_DIR}/src/nn/Linear.cpp
    ${CMAKE_SOURCE_DIR}/src/nn/MLP.cpp
    ${CMAKE_SOURCE_DIR}/src/nn/Loss.cpp
    ${CMAKE_SOURCE_DIR}/src/optim/SGD.cpp
    ${CMAKE_SOURCE_DIR}/src/optim/Adam.cpp
    ${CMAKE_SOURCE_DIR}/src/nn/Conv2D.cpp
    ${CMAKE_SOURCE_DIR}/src/nn/MaxPool2D.cpp
    ${CMAKE_SOURCE_DIR}/src/nn/Sequential.cpp
)
add_test(NAME NNIntegrationTest COMMAND nn_tests)