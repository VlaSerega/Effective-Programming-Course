cmake_minimum_required(VERSION 3.17)
project(Lab_4 C)

SET(GCC_COVERAGE_COMPILE_FLAGS "-march=native -mavx -fopenmp")
set(CMAKE_C_STANDARD 11)
SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")

add_executable(Lab_4 parallel_4line.c)

target_link_libraries(Lab_4 PRIVATE m)