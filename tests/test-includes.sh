#!/bin/sh

set -e

PREFIX=$(pwd)/tests/.test-includes
mkdir -p $PREFIX && cd $PREFIX

CXX=g++

for t in $(../gen-test-prog.py ../../include/crystalnet/**/*); do
	echo $t
	$CXX -I../../include -std=c++1z -c $PREFIX/$t
done
