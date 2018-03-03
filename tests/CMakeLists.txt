cmake_minimum_required(VERSION 3.9)
project(misaka-test)

set(CMAKE_CXX_STANDARD 17)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

add_definitions(-funroll-loops)
add_definitions(-DMISAKA_DEBUG)
add_definitions(-DENABLE_DEBUG_TIMER)

include_directories(${CMAKE_SOURCE_DIR}/../src)
link_directories(${CMAKE_SOURCE_DIR}/../build/lib)

file(GLOB tests *.c *.cpp)
foreach(main ${tests})
    get_filename_component(bin_name ${main} NAME_WE)
    add_executable(${bin_name} ${main})
    target_link_libraries(${bin_name} misaka)
endforeach()
