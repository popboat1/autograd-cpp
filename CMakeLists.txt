cmake_minimum_required(VERSION 3.15)
project(CppMLFramework LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# compiler optim flags
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

if(MSVC)
    add_compile_options(/O2 /Oi /Ot /GL)
else()
    add_compile_options(-O3 -march=native -ffast-math)
endif()

# enable link-time optimization
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)
if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    message(STATUS "Link-Time Optimization (LTO) enabled for Release builds")
endif()

# multi threading with openmp
find_package(OpenMP REQUIRED)

# silence deprecation warnings triggered by third-party libraries
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)


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
    src/autograd/Tensor.cpp
    src/nn/Linear.cpp
    src/nn/MLP.cpp
    src/optim/SGD.cpp
    src/optim/Adam.cpp
    src/nn/Loss.cpp
    src/nn/Conv2D.cpp
    src/nn/MaxPool2D.cpp
    src/nn/AvgPool2D.cpp
    src/nn/BatchNorm2D.cpp
    src/nn/Sequential.cpp
)
target_link_libraries(autograd_main PRIVATE OpenMP::OpenMP_CXX)

# compile to python library
pybind11_add_module(autograd_cpp 
    src/autograd/bindings.cpp
    src/autograd/Tensor.cpp
    src/nn/Linear.cpp
    src/nn/MLP.cpp
    src/optim/SGD.cpp
    src/optim/Adam.cpp
    src/nn/Loss.cpp
    src/nn/Conv2D.cpp
    src/nn/MaxPool2D.cpp
    src/nn/AvgPool2D.cpp
    src/nn/BatchNorm2D.cpp
    src/nn/Sequential.cpp
)
target_link_options(autograd_cpp PRIVATE "-static-libgcc" "-static-libstdc++")
target_link_libraries(autograd_cpp PRIVATE OpenMP::OpenMP_CXX)

set_target_properties(autograd_cpp PROPERTIES 
    PREFIX ""
    SUFFIX ".pyd"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

enable_testing()
add_subdirectory(tests)