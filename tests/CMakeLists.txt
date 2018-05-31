cmake_minimum_required(VERSION 3.9)
project(crystalnet-test)

set(CMAKE_CXX_STANDARD 17)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

add_definitions(-funroll-loops)
add_definitions(-DCRYSTALNET_DEBUG)
add_definitions(-DENABLE_DEBUG_TIMER)

include_directories(${CMAKE_SOURCE_DIR}/../include)
link_directories(${CMAKE_SOURCE_DIR}/../build/${CMAKE_SYSTEM_NAME}/lib)

file(GLOB tests src/crystalnet/**/*)
foreach(main ${tests})
    get_filename_component(base_name ${main} NAME)
    string(REPLACE "." "_" bin_name ${base_name})
    add_executable(${bin_name} ${main})
    target_link_libraries(${bin_name} crystalnet)
endforeach()
