cmake_minimum_required(VERSION 3.16.3)
project(OpenCV_face_detection_demo VERSION 1.0.0 LANGUAGES CXX)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS}) # Not needed for CMake >= 2.8.11

# several variables will be set by OpenCVConfig.cmake
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "    install path: ${OpenCV_INSTALL_PATH}")

add_executable(OpenCV_demo.out ${CMAKE_CURRENT_SOURCE_DIR}/src/face_detection.cpp)
target_link_libraries(OpenCV_demo.out ${OpenCV_LIBS})  # or "OpenCV_LIBRARIES"