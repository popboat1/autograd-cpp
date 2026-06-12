add_executable(autograd_tests
    autograd_tests.cpp
    ${CMAKE_SOURCE_DIR}/src/autograd/Value.cpp
    ${CMAKE_SOURCE_DIR}/src/nn/Linear.cpp
)

add_test(NAME AutogradTest COMMAND autograd_tests)