cmake_minimum_required(VERSION 3.5.1)
project(3d_fluid)

find_package(Celerity CONFIG REQUIRED)

add_executable(3d_fluid 3d_fluid.cpp)
add_celerity_to_target(TARGET 3d_fluid SOURCES 3d_fluid.cpp)