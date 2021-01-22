cmake_minimum_required(VERSION 3.5.1)
project(3d_stencil)

find_package(Celerity CONFIG REQUIRED)

add_executable(3d_stencil 3d_stencil.cpp)
add_celerity_to_target(TARGET 3d_stencil SOURCES 3d_stencil.cpp)