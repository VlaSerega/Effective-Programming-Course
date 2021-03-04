cmake_minimum_required(VERSION 3.17)
project(Lab_2 C)

SET(GCC_COVERAGE_COMPILE_FLAGS "-march=native -O3")
set(CMAKE_C_STANDARD 11)
SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")

add_executable(Lab_2 My_version.c)

target_link_libraries(Lab_2 PRIVATE m)