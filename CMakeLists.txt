cmake_minimum_required(VERSION 3.9)
project(crystalnet)

if ((${CMAKE_SYSTEM_NAME} MATCHES "Darwin"))
    set(USE_OPENBLAS on)
endif()
set(CMAKE_CXX_STANDARD 17)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# add_definitions(-funroll-loops)
if (${USE_OPENBLAS})
    add_definitions(-DCRYSTALNET_USE_CBLAS)
    set(OPENBLAS_PATH $ENV{HOME}/local/openblas)
    include_directories(${OPENBLAS_PATH}/include)
    link_directories(${OPENBLAS_PATH}/lib)
endif()

include_directories(${CMAKE_SOURCE_DIR}/src)
link_directories(${LIBRARY_OUTPUT_PATH})

file(GLOB srcs src/crystalnet/**/*.cpp)
add_library(crystalnet SHARED ${srcs})
if (USE_OPENBLAS)
    target_link_libraries(crystalnet openblas)
endif()

file(GLOB examples examples/c/*.c)
foreach(main ${examples})
    get_filename_component(base_name ${main} NAME)
    string(REPLACE "." "_" bin_name ${base_name})
    add_executable(${bin_name} ${main})
    target_link_libraries(${bin_name} crystalnet)
endforeach()

set(CMAKE_INSTALL_PREFIX $ENV{HOME}/local/crystalnet)  # TODO: make it configurable
install(TARGETS crystalnet LIBRARY DESTINATION lib)
file(GLOB headers src/*.h)
install(FILES ${headers} DESTINATION include)


add_custom_target(doxygen COMMAND doxygen ${CMAKE_SOURCE_DIR}/docs/Doxyfile)
# add_custom_target(sphinx-doc COMMAND sphinx-build -b html html sphinx/build/html
#                   DEPENDS doxygen)


# contrib
find_package(opencv)

file(GLOB yolo_srcs src/crystalnet-contrib/**/*.cpp
                    src/crystalnet-contrib/**/*.c)
add_executable(yolo ${yolo_srcs})
target_link_libraries(yolo crystalnet)
IF("${CMAKE_SYSTEM}" MATCHES "Linux")
    target_link_libraries(yolo stdc++fs)
endif()

target_link_libraries(yolo opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)


# https://gitlab.kitware.com/cmake/community/wikis/doc/cpack/PackageGenerators
SET(CPACK_GENERATOR "TGZ")
# if ((${CMAKE_SYSTEM_NAME} MATCHES "Linux"))
    # SET(CPACK_GENERATOR "TGZ;DEB")
    # SET(CPACK_GENERATOR "TGZ;RPM")
# endif()
INCLUDE(CPack)
