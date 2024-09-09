cmake_minimum_required(VERSION 3.16.3)
project(OpenCV_face_detection_demo VERSION 1.0.0 LANGUAGES CXX)

find_package(OpenCV)
include_directories(OpenCV_INCLUDE_DIRS)
message(${OpenCV_INCLUDE_DIRS})
message(${OpenCV_LIBRARIES})

add_executable(OpenCV_demo.out ${CMAKE_CURRENT_SOURCE_DIR}/src/face_detection.cpp)
target_link_libraries(OpenCV_demo.out ${OpenCV_LIBRARIES})