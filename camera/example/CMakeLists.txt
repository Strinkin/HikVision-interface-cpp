cmake_minimum_required(VERSION 3.16)

project(example)

add_definitions(-std=c++17)
set(CMAKE_CXX_STANDARD 17)

# 设置 example 目录下的可执行文件和对象文件输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/example/bin) # 可执行文件
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/example/lib/static) # 静态库
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/example/lib/shared) # 动态库

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_subdirectory(camera)

add_executable(example example.cpp)
target_link_libraries(example PRIVATE camera yaml-cpp cudart ${OpenCV_LIBS} )
target_include_directories(example PRIVATE /usr/local/cuda-11.4/include)
target_link_directories(example PRIVATE /usr/local/cuda-11.4/lib64)