cmake_minimum_required(VERSION 3.14)
project(KNN LANGUAGES CXX)
message(STATUS "compiler = " ${CMAKE_C_COMPILER_ID})

add_executable(knn-openmp knn.cpp)
target_link_options(knn-openmp PRIVATE -fopenmp)
target_compile_options(knn-openmp PRIVATE -fopenmp -Wall -pedantic -march=native)
