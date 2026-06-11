add_executable(autograd_tests
    autograd_tests.cpp
    ../src/autograd/Value.cpp
)

add_test(NAME AutogradTest COMMAND autograd_tests)