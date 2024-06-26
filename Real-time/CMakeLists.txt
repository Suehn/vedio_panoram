


cmake_minimum_required(VERSION 3.10)

# 项目名称
project(RealTimePanoramicStitching)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)

# 包含头文件目录
include_directories(${CMAKE_SOURCE_DIR}/Real-timePanoramicStitching)

# 查找源文件
file(GLOB SOURCES
    "${CMAKE_SOURCE_DIR}/Real-timePanoramicStitching/*.cpp"
    "${CMAKE_SOURCE_DIR}/Real-timePanoramicStitching/*.h"
)

# 添加可执行文件
add_executable(RealTimePanoramicStitching ${SOURCES})


