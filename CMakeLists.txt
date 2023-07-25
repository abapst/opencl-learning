cmake_minimum_required(VERSION 3.0)

project(OpenCL-Test)

set_property(GLOBAL PROPERTY CXX_STANDARD 17)

find_package(OpenCL REQUIRED)

# expects to find prebuilt opencv in local user directory
find_package(OpenCV REQUIRED PATHS $ENV{USERPROFILE}\\opencv\\build)

include_directories(${OpenCL_INCLUDE_DIRS}) 
message(STATUS "OpenCL_INCLUDE_DIRS = ${OpenCL_INCLUDE_DIRS}")
link_directories(${OpenCL_LIBRARY})

include_directories( ${OpenCV_INCLUDE_DIRS} )
message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")

add_executable(OpenCL-Test)

target_sources(
    OpenCL-Test
PRIVATE
    main.cpp
)

target_link_libraries(OpenCL-Test
    ${OpenCL_LIBRARY}
    ${OpenCV_LIBS}
)
