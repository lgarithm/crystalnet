cmake_minimum_required(VERSION 3.9)
project(light)

set(CMAKE_CXX_STANDARD 17)

set(CRYSTALNET_PATH $ENV{HOME}/local/crystalnet)  # make it configurable
include_directories(${CRYSTALNET_PATH}/include)
link_directories(${CRYSTALNET_PATH}/lib)

set(DARKNET_PATH $ENV{HOME}/local/darknet)  # make it configurable
include_directories(${DARKNET_PATH}/include)
link_directories(${DARKNET_PATH}/lib)

file(GLOB srcs src/*.cpp)
add_executable(light ${srcs})
target_link_libraries(light crystalnet darknet)
IF("${CMAKE_SYSTEM}" MATCHES "Linux")
    target_link_libraries(light stdc++fs)
endif()

