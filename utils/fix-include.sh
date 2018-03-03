#!/bin/sh

iwyu -std=c++1z -I./src src/crystalnet/core/tensor.cpp 2>&1 | fix_include -b --comments
