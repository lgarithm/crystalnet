INCLUDE(ExternalProject)

SET(STDNN_OPS_GIT_URL https://github.com/lgarithm/stdnn-ops.git
    CACHE STRING "URL for clone stdtensor")

SET(PREFIX ${CMAKE_SOURCE_DIR}/3rdparty)

EXTERNALPROJECT_ADD(stdnn-ops-repo
                    GIT_REPOSITORY
                    ${STDNN_OPS_GIT_URL}
                    GIT_TAG
                    v0.1.0
                    PREFIX
                    ${PREFIX}
                    CMAKE_ARGS
                    -DCMAKE_INSTALL_PREFIX=${PREFIX}
                    -DBUILD_TESTS=0
                    -DBUILD_EXAMPLES=0
                    -DBUILD_BENCHMARKS=0
                    -DBUILD_PACKAGES=1
                    -DSTDTENSOR_GIT_URL=${STDTENSOR_GIT_URL})
