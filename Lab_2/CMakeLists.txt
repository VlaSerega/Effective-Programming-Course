cmake_minimum_required(VERSION 3.17)
project(Lab_2 C)

set(CMAKE_C_STANDARD 11)

add_executable(Lab_2 main.c main.c)

target_link_libraries(Lab_2 PRIVATE m)