# scalar engine unit tests
add_executable(autograd_tests
    autograd_tests.cpp
    ${CMAKE_SOURCE_DIR}/src/autograd/Value.cpp
    ${CMAKE_SOURCE_DIR}/src/autograd/Tensor.cpp
    ${CMAKE_SOURCE_DIR}/src/nn/Linear.cpp
)
add_test(NAME AutogradScalarTest COMMAND autograd_tests)

# tensor engine unit tests
add_executable(tensorgrad_tests
    tensorgrad_tests.cpp
    ${CMAKE_SOURCE_DIR}/src/autograd/Tensor.cpp
)
add_test(NAME TensorGradMatrixTest COMMAND tensorgrad_tests)