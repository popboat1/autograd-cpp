cmake_minimum_required(VERSION 3.15)
project(CppMLFramework LANGUAGES CXX)

# silence deprecation warnings triggered by third-party libraries
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/src)

# find venv dynamically
if(DEFINED ENV{VIRTUAL_ENV})
    set(Python3_EXECUTABLE "$ENV{VIRTUAL_ENV}/Scripts/python.exe")
    set(Python3_FIND_VIRTUALENV ONLY)
    message(STATUS "Targeting Virtual Environment Python: ${Python3_EXECUTABLE}")
endif()

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}" CACHE INTERNAL "")

# fetch pybind 11.0
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.1
)
FetchContent_MakeAvailable(pybind11)

add_executable(autograd_main
    src/main.cpp
    src/autograd/Value.cpp
    src/nn/Linear.cpp
    src/nn/MLP.cpp
    src/optim/SGD.cpp
    src/nn/Loss.cpp
    src/autograd/Tensor.cpp
)

# compile to python library
pybind11_add_module(autograd_cpp 
    src/autograd/Value.cpp
    src/autograd/bindings.cpp
    src/nn/Linear.cpp
    src/nn/MLP.cpp
    src/optim/SGD.cpp
    src/nn/Loss.cpp
    src/autograd/Tensor.cpp
)

target_link_options(autograd_cpp PRIVATE "-static-libgcc" "-static-libstdc++")

set_target_properties(autograd_cpp PROPERTIES 
    PREFIX ""
    SUFFIX ".pyd"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

enable_testing()
add_subdirectory(tests)