#!/bin/sh

set -e

cd $(dirname $0)/..
ROOT=$(pwd)

PREFIX=${ROOT}/tests/.test-includes
mkdir -p $PREFIX && cd $PREFIX

CXX=g++

INCLUDES="-I${ROOT}/include -I${ROOT}/3rdparty/include"
for t in $(../gen-test-prog.py ../../include/crystalnet/**/*); do
    echo $t
    $CXX ${INCLUDES} -std=c++1z -c $PREFIX/$t
done
