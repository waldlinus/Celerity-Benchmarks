cmake_minimum_required(VERSION 3.5.1)
project(2d_stencil)

find_package(Celerity CONFIG REQUIRED)

add_executable(2d_stencil 2d_stencil.cpp)
add_celerity_to_target(TARGET 2d_stencil SOURCES 2d_stencil.cpp)