cmake_minimum_required(VERSION 3.17)
project(Lab_1 C)

set(CMAKE_C_STANDARD 11)

add_executable(Lab_1 main.c)

target_link_libraries(Lab_1 PRIVATE m)