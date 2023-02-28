cmake_minimum_required(VERSION 3.23)
project(Lab6)

set(CMAKE_CXX_STANDARD 14)

find_package(MPI)
find_package(OpenMP)
include_directories(${MPI_INCLUDE_PATH})

add_executable(Lab6 main.cpp)
