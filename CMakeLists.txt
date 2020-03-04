cmake_minimum_required (VERSION 3.1.0)
project (pyrol LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-D_USE_MATH_DEFINES)
add_definitions(-DNOMINMAX) # avoid min/max macros in <windows.h>

find_package(RL COMPONENTS MDL SG PLAN KIN REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(pyrol
  pyrol.cpp)

target_link_libraries(pyrol
  PUBLIC
    rl::mdl
    rl::sg
)
