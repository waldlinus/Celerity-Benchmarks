cmake_minimum_required(VERSION 3.5.1)
project(server)

find_package(Celerity CONFIG REQUIRED)

add_executable(server server.cpp)
add_celerity_to_target(TARGET server SOURCES server.cpp)