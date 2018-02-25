cmake_minimum_required(VERSION 3.10)
project(misaka)

set(CMAKE_CXX_STANDARD 17)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

add_definitions(-funroll-loops)
# add_definitions(-DMISAKA_DEBUG)  # TODO: make it optional
add_definitions(-DENABLE_DEBUG_TIMER)  # TODO: make it optional

include_directories(${CMAKE_SOURCE_DIR}/src)
link_directories(${LIBRARY_OUTPUT_PATH})

# find_package(Threads REQUIRED)

file(GLOB srcs src/misaka/**/*.cpp src/teavana/*.cpp)
file(GLOB tests src/misaka/**/*_test.*)
list(REMOVE_ITEM srcs ${tests})
add_library(misaka SHARED ${srcs})

file(GLOB tests src/misaka/**/*_test.*)
foreach(main ${tests})
    get_filename_component(bin_name ${main} NAME_WE)
    add_executable(${bin_name} ${main})
    target_link_libraries(${bin_name} misaka)
    # target_link_libraries(${bin_name} openblas)
endforeach()

file(GLOB examples examples/c/*.c)
foreach(main ${examples})
    get_filename_component(bin_name ${main} NAME_WE)
    add_executable(${bin_name}_c ${main})
    target_link_libraries(${bin_name}_c misaka)
    # target_link_libraries(${bin_name}_c openblas)
endforeach()

file(GLOB examples examples/cpp/*.cpp)
foreach(main ${examples})
    get_filename_component(bin_name ${main} NAME_WE)
    add_executable(${bin_name}_cpp ${main})
    target_link_libraries(${bin_name}_cpp misaka)
    # target_link_libraries(${bin_name}_cpp openblas)
endforeach()

set(CMAKE_INSTALL_PREFIX $ENV{HOME}/local/misaka)  # TODO: make it configurable
install(TARGETS misaka LIBRARY DESTINATION lib)
