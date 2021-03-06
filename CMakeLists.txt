cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

# for argument parser
set(ARGPARSER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/argparse")
# include_directories(${ARGPARSER_DIR})
set(ARGPARSER_SRC "${ARGPARSER_DIR}/argparse.c")
add_library(arg_parser ${ARGPARSER_SRC})
target_include_directories(arg_parser PUBLIC ${ARGPARSER_DIR})

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

project(camera_fusion)

find_package(OpenCV 4.1 REQUIRED)
find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

option(WITH_CUDA off)

if (WITH_CUDA)
  add_compile_definitions(WITH_CUDA)
endif()

# Executable for create matrix exercise
add_executable (3D_object_tracking src/camFusion_Student.cpp src/FinalProject_Camera.cpp src/lidarData.cpp src/matching2D_Student.cpp src/objectDetection2D.cpp)
target_link_libraries (3D_object_tracking ${OpenCV_LIBRARIES} arg_parser ${PCL_LIBRARIES})