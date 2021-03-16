cmake_minimum_required(VERSION 3.17)
project(Lab_3 C)

SET(GCC_COVERAGE_COMPILE_FLAGS "-march=native -mavx")
set(CMAKE_C_STANDARD 11)
SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")

add_executable(Lab_3 inline_asm_3line.c)

target_link_libraries(Lab_3 PRIVATE m)