cmake_minimum_required(VERSION 3.9)
project(tiny-imagenet)

set(CMAKE_CXX_STANDARD 17)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

set(OPENCV_PATH $ENV{HOME}/local/opencv)
include_directories(${OPENCV_PATH}/include)
link_directories(${OPENCV_PATH}/lib)

set(INSTALL_PREFIX $ENV{HOME}/local/crystalnet)  # make it configurable
include_directories(${INSTALL_PREFIX}/include)
link_directories(${INSTALL_PREFIX}/lib)

file(GLOB srcs src/*.cpp src/*.c)
add_executable(test-imagenet ${srcs})

target_link_libraries(test-imagenet crystalnet)
IF("${CMAKE_SYSTEM}" MATCHES "Linux")
    target_link_libraries(test-imagenet stdc++fs)
endif()
# apt install -y libopencv-dev
target_link_libraries(test-imagenet opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs) 
