#!/bin/sh

set -e

ROOT=$(pwd)

BUILD_DIR=${ROOT}/build/$(uname -s)
LOG_DIR=$(pwd)/log
LOG_FILE=${LOG_DIR}/tidy.log

clang-tidy -version

for SRC in $(find ${ROOT}/src -type f -name '*.cpp'); do
	echo "tidy ${SRC}"
	clang-tidy -p ${BUILD_DIR} -fix -header-filter=.* -system-headers ${SRC} | tee -a ${LOG_FILE}
done
