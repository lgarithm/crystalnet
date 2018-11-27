CMAKE_MINIMUM_REQUIRED(VERSION 3.9)
PROJECT(crystalnet)

IF((${CMAKE_SYSTEM_NAME} MATCHES "Darwin"))
    SET(USE_OPENBLAS on)
ENDIF()
SET(CMAKE_CXX_STANDARD 17)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

INCLUDE(cmake/3rdparty.cmake)

# add_definitions(-funroll-loops)
IF(${USE_OPENBLAS})
    ADD_DEFINITIONS(-DCRYSTALNET_USE_CBLAS)
    SET(OPENBLAS_PATH $ENV{HOME}/local/openblas)
    INCLUDE_DIRECTORIES(${OPENBLAS_PATH}/include)
    LINK_DIRECTORIES(${OPENBLAS_PATH}/lib)
ENDIF()

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include)
LINK_DIRECTORIES(${LIBRARY_OUTPUT_PATH})

FILE(GLOB srcs src/crystalnet/**/*.cpp)
ADD_LIBRARY(crystalnet SHARED ${srcs})
ADD_DEPENDENCIES(crystalnet stdtracer-repo)
TARGET_LINK_LIBRARIES(crystalnet stdtracer)
IF(USE_OPENBLAS)
    TARGET_LINK_LIBRARIES(crystalnet openblas)
ENDIF()

FILE(GLOB examples examples/c/*.c)
FOREACH(main ${examples})
    GET_FILENAME_COMPONENT(base_name ${main} NAME)
    STRING(REPLACE "."
                   "_"
                   bin_name
                   ${base_name})
    ADD_EXECUTABLE(${bin_name} ${main})
    TARGET_LINK_LIBRARIES(${bin_name} crystalnet)
ENDFOREACH()

SET(CMAKE_INSTALL_PREFIX $ENV{HOME}/local/crystalnet) # TODO: make it
                                                      # configurable
INSTALL(TARGETS crystalnet LIBRARY DESTINATION lib)
INSTALL(DIRECTORY include DESTINATION .)

ADD_CUSTOM_TARGET(doxygen COMMAND doxygen ${CMAKE_SOURCE_DIR}/docs/Doxyfile)
# add_custom_target(sphinx-doc COMMAND sphinx-build -b html html
# sphinx/build/html DEPENDS doxygen)

# contrib
FIND_PACKAGE(opencv)

FILE(GLOB yolo_srcs src/crystalnet-contrib/**/*.cpp
          src/crystalnet-contrib/**/*.c)
ADD_EXECUTABLE(yolo ${yolo_srcs})
TARGET_LINK_LIBRARIES(yolo crystalnet)
IF("${CMAKE_SYSTEM}" MATCHES "Linux")
    TARGET_LINK_LIBRARIES(yolo stdc++fs)
ENDIF()

TARGET_LINK_LIBRARIES(yolo
                      opencv_core
                      opencv_imgproc
                      opencv_highgui
                      opencv_imgcodecs)

# https://gitlab.kitware.com/cmake/community/wikis/doc/cpack/PackageGenerators
SET(CPACK_PACKAGE_DIRECTORY ${CMAKE_SOURCE_DIR}/release)
SET(CPACK_GENERATOR "TGZ")
SET(CPACK_PACKAGE_VERSION "latest")
# if ((${CMAKE_SYSTEM_NAME} MATCHES "Linux")) SET(CPACK_GENERATOR "TGZ;DEB")
# SET(CPACK_GENERATOR "TGZ;RPM") endif()
INCLUDE(CPack)
