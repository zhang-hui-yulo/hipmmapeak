cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(hipmmapeak LANGUAGES CXX HIP)

find_package(HIP)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)

if(NOT DEFINED MMA_HIP_ARCHITECTURES)
    set(MMA_HIP_ARCHITECTURES gfx1100;gfx1201)
endif()

add_executable(hipmmapeak
    src/hipmmapeak.hip)

string(REPLACE ";" " " REQUIRED_HIP_ARCHITECTURES "${MMA_HIP_ARCHITECTURES}")

target_compile_definitions(hipmmapeak PRIVATE
    REQUIRED_HIP_ARCHITECTURES="${REQUIRED_HIP_ARCHITECTURES}")

set_target_properties(hipmmapeak PROPERTIES
    HIP_ARCHITECTURES "${MMA_HIP_ARCHITECTURES}")

