cmake_minimum_required(VERSION 3.16)

project(camera)

add_definitions(-std=c++17)
set(CMAKE_CXX_STANDARD 17)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Project Source file
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp)

# 设置库文件的输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/static) # 静态库
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/shared) # 动态库

# add library
add_library(camera SHARED ${SRCS})

# HikVision SDK
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("embed_platform on")
  # include_directories(${PROJECT_SOURCE_DIR}/SDK/HikVision/aarch64/include)
  link_directories(${PROJECT_SOURCE_DIR}/SDK/aarch64/lib)
  target_link_directories(camera PUBLIC ${PROJECT_SOURCE_DIR}/SDK/HikVision/aarch64/lib) # HikVision lib
else()
  message("embed_platform off")
  include_directories(${PROJECT_SOURCE_DIR}/SDK/HikVision/x86/include)
  # link_directories(${PROJECT_SOURCE_DIR}/SDK/HikVision/x86/lib)
  target_link_directories(camera PUBLIC ${PROJECT_SOURCE_DIR}/SDK/HikVision/x86/lib) # HikVision lib
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("embed_platform on")
  set(CMAKE_CUDA_ARCHITECTURES 72)
  target_include_directories(camera PRIVATE /usr/local/cuda/targets/aarch64-linux/include)
  target_link_directories(camera PRIVATE /usr/local/cuda/targets/aarch64-linux/lib)
else()
  message("embed_platform off")
  set(CMAKE_CUDA_ARCHITECTURES 86)
  # cuda
  target_include_directories(camera PRIVATE /usr/local/cuda/include)
  target_link_directories(camera PRIVATE /usr/local/cuda/lib64)
endif()

target_include_directories(camera PUBLIC ${PROJECT_SOURCE_DIR}/SDK/HikVision/x86/include) # HikVision Head file
target_include_directories(camera PUBLIC ${PROJECT_SOURCE_DIR}/include) # Projetc Heade file
target_link_libraries(camera PUBLIC MvCameraControl) 
target_link_libraries(camera PRIVATE yaml-cpp ${OpenCV_LIBS}) 
