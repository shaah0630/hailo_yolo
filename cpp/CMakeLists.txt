cmake_minimum_required(VERSION 3.16)
project(HailoYolo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")

add_executable(detect_image detect_image.cpp inference/hailo.cpp)

target_include_directories(detect_image PRIVATE inference/)
target_link_libraries(detect_image PRIVATE ${OpenCV_LIBS} -lhailort)