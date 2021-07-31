cmake_minimum_required(VERSION 3.1.0)

project(VPPatternGenerator)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(VPPatternGenerator VPPatternGenerator.cpp)

target_link_libraries(VPPatternGenerator ${OpenCV_LIBS})
