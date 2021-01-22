cmake_minimum_required(VERSION 3.5.1)
project(1d_stencil)

find_package(Celerity CONFIG REQUIRED)

add_executable(1d_stencil 1d_stencil.cpp)
add_celerity_to_target(TARGET 1d_stencil SOURCES 1d_stencil.cpp)