cmake_minimum_required(VERSION 3.9)
project(crystalnet)

set(USE_OPENBLAS off)
set(CMAKE_CXX_STANDARD 17)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

add_definitions(-funroll-loops)
# add_definitions(-DCRYSTALNET_DEBUG)  # TODO: make it optional
add_definitions(-DENABLE_DEBUG_TIMER)  # TODO: make it optional
if (${USE_OPENBLAS})
    add_definitions(-DCRYSTALNET_USE_CBLAS)
endif()

include_directories(${CMAKE_SOURCE_DIR}/src)
link_directories(${LIBRARY_OUTPUT_PATH})

file(GLOB srcs src/crystalnet/**/*.cpp src/teavana/*.cpp)
add_library(crystalnet SHARED ${srcs})
if (USE_OPENBLAS)
    target_link_libraries(crystalnet openblas)
endif()

file(GLOB examples examples/c/*.c)
foreach(main ${examples})
    get_filename_component(bin_name ${main} NAME_WE)
    add_executable(${bin_name}_c ${main})
    target_link_libraries(${bin_name}_c crystalnet)
endforeach()

file(GLOB examples examples/cpp/*.cpp)
foreach(main ${examples})
    get_filename_component(bin_name ${main} NAME_WE)
    add_executable(${bin_name}_cpp ${main})
    target_link_libraries(${bin_name}_cpp crystalnet)
endforeach()

set(CMAKE_INSTALL_PREFIX $ENV{HOME}/local/crystalnet)  # TODO: make it configurable
install(TARGETS crystalnet LIBRARY DESTINATION lib)
