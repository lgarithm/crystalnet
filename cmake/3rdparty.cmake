
INCLUDE(ExternalProject)

SET(PREFIX ${CMAKE_SOURCE_DIR}/3rdparty)

EXTERNALPROJECT_ADD(libstdtracer
                    GIT_REPOSITORY
                    https://github.com/lgarithm/stdtracer
                    GIT_TAG
                    05c6d71afd272b6008504fce66099877c4cec1ef
                    PREFIX
                    ${PREFIX}
                    CMAKE_ARGS
                    -DCMAKE_INSTALL_PREFIX=${PREFIX}
                    -DCMAKE_CXX_FLAGS=-fPIC) # TODO:  fix in upstream

EXTERNALPROJECT_ADD(libstdtensor
                    GIT_REPOSITORY
                    https://github.com/lgarithm/stdtensor
                    GIT_TAG
                    master
                    PREFIX
                    ${PREFIX}
                    CMAKE_ARGS
                    -DCMAKE_INSTALL_PREFIX=${PREFIX}
                    -DBUILD_TESTS=0
                    -DBUILD_EXAMPLES=0)

EXTERNALPROJECT_ADD(libstdnn-ops
                    GIT_REPOSITORY
                    https://github.com/lgarithm/stdnn-ops
                    GIT_TAG
                    dev
                    PREFIX
                    ${PREFIX}
                    CMAKE_ARGS
                    -DCMAKE_INSTALL_PREFIX=${PREFIX}
                    -DBUILD_TESTS=0
                    -DBUILD_EXAMPLES=0
                    -DBUILD_PACKAGES=1)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/3rdparty/include)
LINK_DIRECTORIES(${CMAKE_SOURCE_DIR}/3rdparty/lib)
