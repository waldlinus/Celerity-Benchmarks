cmake_minimum_required(VERSION 3.5.1)
project(crystal)

find_package(Celerity CONFIG REQUIRED)

add_executable(crystal crystal.cpp)
add_celerity_to_target(TARGET crystal SOURCES crystal.cpp)