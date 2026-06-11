cmake_minimum_required(VERSION 3.15)
project(CppMLFramework LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(src)

add_executable(autograd_main
    src/main.cpp
    src/autograd/Value.cpp
)