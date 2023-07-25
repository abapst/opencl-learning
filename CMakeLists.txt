cmake_minimum_required(VERSION 3.0)

project(OpenCL-Test)

set_property(GLOBAL PROPERTY CXX_STANDARD 17)

FIND_PACKAGE(OpenCL REQUIRED)

INCLUDE_DIRECTORIES(${OpenCL_INCLUDE_DIRS}) 
message(STATUS "OpenCL_INCLUDE_DIRS = ${OpenCL_INCLUDE_DIRS}")

LINK_DIRECTORIES(${OpenCL_LIBRARY})

add_executable(OpenCL-Test)

target_sources(
    OpenCL-Test
PRIVATE
    main.cpp
)

target_link_libraries(OpenCL-Test
    ${OpenCL_LIBRARY}
)
