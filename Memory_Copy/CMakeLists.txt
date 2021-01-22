cmake_minimum_required(VERSION 3.5.1)
project(mem_copy)

find_package(Celerity CONFIG REQUIRED)

add_executable(mem_copy mem_copy.cpp)
add_celerity_to_target(TARGET mem_copy SOURCES mem_copy.cpp)