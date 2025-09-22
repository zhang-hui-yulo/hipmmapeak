cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(hipmmapeak LANGUAGES CXX HIP)

find_package(HIP)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)

if(NOT DEFINED MMA_HIP_ARCHITECTURES)
    set(MMA_HIP_ARCHITECTURES gfx1100;gfx1201;gfx1036)
endif()

add_executable(hipmmapeak
    hipmmapeak.hip)

set_target_properties(hipmmapeak PROPERTIES
    HIP_ARCHITECTURES "${MMA_HIP_ARCHITECTURES}")
